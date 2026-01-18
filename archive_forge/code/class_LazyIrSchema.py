from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
class LazyIrSchema:
    name: 'OperatorName'
    positional_args: Tuple[LazyArgument, ...]
    keyword_args: Tuple[LazyArgument, ...]
    returns: Tuple['Return', ...]
    generator_arg: Optional[NamedCType] = None
    func: FunctionSchema
    symint: bool
    properties: LazyIrProperties = LazyIrProperties('ShapePrecompute', 'Lower', 'CanBeReused')
    opkind: Optional[str] = None

    def __init__(self, func: FunctionSchema, properties: Optional[LazyIrProperties]=None, *, symint: bool):
        if properties:
            self.properties = properties
        self.func = func
        self.symint = symint
        positional_args: List[LazyArgument] = []
        for arg_field in ['pre_self_positional', 'self_arg', 'post_self_positional']:
            if arg_field == 'self_arg' and func.arguments.self_arg is not None:
                arg = func.arguments.self_arg.argument
                positional_args.append(LazyArgument(arg, self.properties, symint=symint))
            elif getattr(func.arguments, arg_field) is not None:
                positional_args.extend((LazyArgument(arg, self.properties, symint=symint) for arg in getattr(func.arguments, arg_field)))
        self.positional_args = tuple(positional_args)
        keyword_args: List[LazyArgument] = []
        for arg_field in ['pre_tensor_options_kwarg_only', 'tensor_options', 'post_tensor_options_kwarg_only', 'out']:
            curr_args = getattr(func.arguments, arg_field)
            if curr_args is not None:
                if isinstance(curr_args, TensorOptionsArguments):
                    curr_args = curr_args.all()
                for arg in curr_args:
                    if isGeneratorType(arg.type):
                        assert self.generator_arg is None, 'We expect there is only one generator arg'
                        self.generator_arg = NamedCType(arg.name, arg.type)
                keyword_args.extend((LazyArgument(arg, self.properties, symint=symint) for arg in curr_args))
        self.keyword_args = tuple(keyword_args)
        self.name = func.name
        self.returns = func.returns

    @property
    def node_name(self) -> str:
        """
        Return camel-case version of op in node.

        Note: This function also appends any `overload_name` in the operation.
        For example, if the op is `bitwise_and.Tensor`, the returned name
        will be `BitwiseAndTensor`.
        """
        op_name = f'{self.name.name}_{self.name.overload_name}'.lower()
        return ''.join((word.capitalize() or '' for word in op_name.split('_')))

    @property
    def aten_name(self) -> str:
        return str(self.name.name)

    @property
    def base_name(self) -> str:
        return f'{self.name.name.base}'

    def filtered_args(self, positional: bool=True, keyword: bool=True, values: bool=True, scalars: bool=True, generator: bool=True) -> List[LazyArgument]:
        args: List[LazyArgument] = []
        if positional:
            args.extend(self.positional_args)
        if keyword:
            args.extend(self.keyword_args)
        if values and scalars and generator:
            return args
        elif values and scalars:
            return [a for a in args if not a.is_generator]
        elif values:
            return [a for a in args if a.is_lazy_value]
        elif scalars:
            return [a for a in args if not a.is_lazy_value and (generator or not a.is_generator)]
        return []

    @property
    def positional_values(self) -> List[LazyArgument]:
        return self.filtered_args(positional=True, keyword=False, values=True, scalars=False)

    @property
    def positional_scalars(self) -> List[LazyArgument]:
        return self.filtered_args(positional=True, keyword=False, values=False, scalars=True)

    @property
    def keyword_values(self) -> List[LazyArgument]:
        return self.filtered_args(positional=False, keyword=True, values=True, scalars=False)

    @property
    def keyword_scalars(self) -> List[LazyArgument]:
        return self.filtered_args(positional=False, keyword=True, values=False, scalars=True)