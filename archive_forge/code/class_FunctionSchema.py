import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class FunctionSchema:
    name: 'OperatorName'
    arguments: 'Arguments'
    returns: Tuple['Return', ...]

    def schema_order_arguments(self) -> Iterator['Argument']:
        return itertools.chain(self.arguments.flat_positional, self.arguments.flat_kwarg_only, self.arguments.out)
    decl_re = re.compile('(?P<name>[^\\(]+)\\((?P<args>.*)\\) -> (?P<returns>.*)')

    @staticmethod
    def parse(func: str) -> 'FunctionSchema':
        decls = FunctionSchema.decl_re.findall(func)
        assert len(decls) == 1, f'Invalid function schema: {func}'
        ops, args, return_decl = decls[0]
        name = OperatorName.parse(ops)
        arguments = Arguments.parse(args)
        returns = parse_returns(return_decl)
        r = FunctionSchema(name=name, arguments=arguments, returns=returns)
        assert str(r) == func, f'{str(r)} != {func}'
        return r

    def returns_are_aliased(self) -> bool:
        return any((r for r in self.returns if r.annotation is not None and r.annotation.is_write))

    def __post_init__(self) -> None:
        for arg, ret in zip(self.arguments.out, self.returns):
            assert arg.annotation == ret.annotation, 'Out arguments must have matching return Tensor; furthermore, the ith-argument needs to correspond to the ith return'
        for a in self.arguments.post_self_positional_mutable:
            assert not any((a.annotation == r.annotation for r in self.returns)), f'If you have a schema with mutable positional args, we expect them to not be returned. schema: {str(self)}'
        out_and_self = list(self.arguments.out) + [arg for arg in self.arguments.flat_positional if arg.name == 'self']
        mutable_returns = [ret for ret in self.returns if ret.annotation is not None and ret.annotation.is_write]
        immutable_returns = [ret for ret in self.returns if ret.annotation is None or not ret.annotation.is_write]
        assert len(mutable_returns) == 0 or len(immutable_returns) == 0, f'NativeFunctions must have either only mutable returns, or only immutable returns. Found: {str(self)}'
        for ret in mutable_returns:
            assert any((ret.annotation == arg.annotation for arg in out_and_self)), 'All mutable returns must be aliased either to a keyword argument, or to "self". Did you forget to mark an out argument as keyword-only?'
        if self.arguments.out:
            if any((a.type != BaseType(BaseTy.Tensor) for a in self.arguments.out)):
                assert len(self.returns) == 0, 'out= ops that accept tensor lists as out arguments '
                "are expected to have no return type (since you can't do method chaining on them)"
            else:
                assert len([arg for arg in self.arguments.out if not arg.name.startswith('_scratch_')]) == len(self.returns), 'Must return as many arguments as there are out arguments, or no return at all'
        if self.name.name.inplace:
            self_a = self.arguments.self_arg
            assert self_a and self_a.argument.annotation and self_a.argument.annotation.is_write
            if self_a.argument.type == BaseType(BaseTy.Tensor):
                assert len(self.returns) == 1 and self.returns[0].annotation == self_a.argument.annotation
            else:
                assert len(self.returns) == 0
        if self.arguments.tensor_options is not None:
            assert self.kind() == SchemaKind.functional, f'Found an operator that is not functional or out variant, but has tensor options arguments.This is not allowed- tensor options arguments are only allowed for factory functions.schema: {str(self)}'
        if self.is_functional_fn():
            assert self.kind() == SchemaKind.functional, f"Found an operator that is not functional, but its overload contains the string 'functional'.This is a special keyword in the codegen, please use a different overload name.schema: {str(self)}"

    def is_functional_fn(self) -> bool:
        return 'functional' in self.name.overload_name

    def is_out_fn(self) -> bool:
        return bool(self.arguments.out)

    def kind(self) -> SchemaKind:
        """
        What kind of schema is this?  A functional schema is one
        that returns a newly allocated output; an inplace schema
        modifies the self argument inplace; an out schema writes
        the result into an explicitly provided out argument.
        """
        is_out = bool(self.arguments.out)
        is_scratch = bool([arg for arg in self.arguments.out if arg.name.startswith('_scratch_')])
        is_inplace = self.name.name.inplace
        is_mutable = any((a.annotation is not None and a.annotation.is_write for a in self.arguments.post_self_positional))
        assert not (is_out and is_inplace)
        if is_inplace:
            return SchemaKind.inplace
        elif is_scratch:
            assert is_out, 'invariant: all scratch operators are expected to be out= operators too'
            return SchemaKind.scratch
        elif is_out:
            assert not is_scratch, 'We should not categorize a scratch op as an out variant. Check if the order of if statements are expected!'
            return SchemaKind.out
        elif is_mutable:
            return SchemaKind.mutable
        else:
            return SchemaKind.functional

    def aliased_return_names(self) -> List[Optional[str]]:
        outs: List[Optional[str]] = []
        for r in self.returns:
            aliased_args = [a for a in self.arguments.flat_all if a.annotation is not None and a.annotation == r.annotation]
            if len(aliased_args) == 0:
                outs.append(None)
            elif len(aliased_args) == 1:
                outs.append(aliased_args[0].name)
            else:
                aliased_names = ', '.join((a.name for a in aliased_args))
                raise AssertionError(f'Found a return ({r.name})that aliases multiple inputs ({aliased_names})')
        return outs

    def signature(self, *, strip_default: bool=False, strip_view_copy_name: bool=False, keep_return_names: bool=False) -> 'FunctionSchema':
        """
                Certain schemas are 'related', in that they are simply
                inplace/out/functional versions of the same function.  This method
                factors these schemas into the "core" functional signature which
                is equal across all versions.

                Here is what normalization happens to the schema to convert
                it to a signature:
                - The overload name is stripped (name is retained, since
                  it expresses semantic content about what the function does)
                - Inplace is set False
                - Out arguments are stripped
                - Mutable post_self_positional args are converted to returns
                - Mutability annotations are stripped  (this is sound
                  because you cannot overload on mutability annotation)
                - Return names are stripped since they are not overloadable and
                  some variants have return names but some not
                - TensorOptions are dropped
                  because out= variants of factory functions don't include them
                  (and we want to be able to pair up factory functions with their out variants)

                Finally, we want to be able to pair up related "view" and their
                corresponding "view_copy" operators. We do this by optionally
                stripping the trailing "_copy" from the base name.

                Example of a mutable op before and after:

                f.func (Mutable operator):
        _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)  # noqa: B950

                f.func (Corresponding functional operator):
        _fused_moving_avg_obs_fq_helper.functional(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask, Tensor running_min_out, Tensor running_max_out, Tensor scale_out, Tensor zero_point_out)  # noqa: B950

                f.func.signature() output:
        _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)  # noqa: B950
        """

        def strip_ret_annotation(r: Return) -> Return:
            return Return(name=r.name if keep_return_names else None, type=r.type, annotation=None)
        base_name = self.name.name.base
        if strip_view_copy_name and base_name.endswith('_copy'):
            base_name = base_name.replace('_copy', '')
        returns_from_mutable_inputs = tuple((Return(name=f'{a.name}_out' if keep_return_names else None, type=a.type, annotation=None) for a in itertools.chain([self.arguments.self_arg.argument] if self.arguments.self_arg is not None else [], self.arguments.out, self.arguments.post_self_positional) if a.annotation is not None and a.annotation.is_write and (not any((a.annotation == r.annotation for r in self.returns)))))
        original_returns = tuple(map(strip_ret_annotation, self.returns))
        returns = original_returns + returns_from_mutable_inputs
        args_sig = self.arguments.signature(strip_default=strip_default)
        if str(self.name) == 'bernoulli.p':
            args_sig = Arguments.parse(str(args_sig).replace('float p', 'float p=0.5'))
        return FunctionSchema(name=OperatorName(name=BaseOperatorName(base=base_name, inplace=False, dunder_method=self.name.name.dunder_method), overload_name=''), arguments=args_sig, returns=returns)

    def view_signature(self) -> 'FunctionSchema':
        return self.signature(strip_view_copy_name=True)

    def with_name(self, name: 'OperatorName') -> 'FunctionSchema':
        return FunctionSchema(name=name, arguments=self.arguments, returns=self.returns)

    @property
    def modifies_arguments(self) -> bool:
        return self.kind() in [SchemaKind.inplace, SchemaKind.out, SchemaKind.mutable]

    def has_symint(self) -> bool:
        return self.arguments.has_symint_arg()

    def __str__(self) -> str:
        all_arguments_str = str(self.arguments)
        if len(self.returns) == 1:
            returns = str(self.returns[0])
        else:
            returns = '(' + ', '.join(map(str, self.returns)) + ')'
        return f'{self.name}({all_arguments_str}) -> {returns}'