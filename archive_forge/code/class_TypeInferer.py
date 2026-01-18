import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
class TypeInferer(object):
    """
    Operates on block that shares the same ir.Scope.
    """

    def __init__(self, context, func_ir, warnings):
        self.context = context
        self.blocks = OrderedDict()
        for k in sorted(func_ir.blocks.keys()):
            self.blocks[k] = func_ir.blocks[k]
        self.generator_info = func_ir.generator_info
        self.func_id = func_ir.func_id
        self.func_ir = func_ir
        self.typevars = TypeVarMap()
        self.typevars.set_context(context)
        self.constraints = ConstraintNetwork()
        self.warnings = warnings
        self.arg_names = {}
        self.assumed_immutables = set()
        self.calls = []
        self.calltypes = utils.UniqueDict()
        self.refine_map = {}
        if config.DEBUG or config.DEBUG_TYPEINFER:
            self.debug = TypeInferDebug(self)
        else:
            self.debug = NullDebug()
        self._skip_recursion = False

    def copy(self, skip_recursion=False):
        clone = TypeInferer(self.context, self.func_ir, self.warnings)
        clone.arg_names = self.arg_names.copy()
        clone._skip_recursion = skip_recursion
        for k, v in self.typevars.items():
            if not v.locked and v.defined:
                clone.typevars[k].add_type(v.getone(), loc=v.define_loc)
        return clone

    def _mangle_arg_name(self, name):
        return 'arg.%s' % (name,)

    def _get_return_vars(self):
        rets = []
        for blk in self.blocks.values():
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                rets.append(inst.value)
        return rets

    def get_argument_types(self):
        return [self.typevars[k].getone() for k in self.arg_names.values()]

    def seed_argument(self, name, index, typ):
        name = self._mangle_arg_name(name)
        self.seed_type(name, typ)
        self.arg_names[index] = name

    def seed_type(self, name, typ):
        """All arguments should be seeded.
        """
        self.lock_type(name, typ, loc=None)

    def seed_return(self, typ):
        """Seeding of return value is optional.
        """
        for var in self._get_return_vars():
            self.lock_type(var.name, typ, loc=None)

    def build_constraint(self):
        for blk in self.blocks.values():
            for inst in blk.body:
                self.constrain_statement(inst)

    def return_types_from_partial(self):
        """
        Resume type inference partially to deduce the return type.
        Note: No side-effect to `self`.

        Returns the inferred return type or None if it cannot deduce the return
        type.
        """
        cloned = self.copy(skip_recursion=True)
        cloned.build_constraint()
        cloned.propagate(raise_errors=False)
        rettypes = set()
        for retvar in cloned._get_return_vars():
            if retvar.name in cloned.typevars:
                typevar = cloned.typevars[retvar.name]
                if typevar and typevar.defined:
                    rettypes.add(types.unliteral(typevar.getone()))
        if not rettypes:
            return
        return cloned._unify_return_types(rettypes)

    def propagate(self, raise_errors=True):
        newtoken = self.get_state_token()
        oldtoken = None
        while newtoken != oldtoken:
            self.debug.propagate_started()
            oldtoken = newtoken
            errors = self.constraints.propagate(self)
            newtoken = self.get_state_token()
            self.debug.propagate_finished()
        if errors:
            if raise_errors:
                force_lit_args = [e for e in errors if isinstance(e, ForceLiteralArg)]
                if not force_lit_args:
                    raise errors[0]
                else:
                    raise reduce(operator.or_, force_lit_args)
            else:
                return errors

    def add_type(self, var, tp, loc, unless_locked=False):
        assert isinstance(var, str), type(var)
        tv = self.typevars[var]
        if unless_locked and tv.locked:
            return
        oldty = tv.type
        unified = tv.add_type(tp, loc=loc)
        if unified != oldty:
            self.propagate_refined_type(var, unified)

    def add_calltype(self, inst, signature):
        assert signature is not None
        self.calltypes[inst] = signature

    def copy_type(self, src_var, dest_var, loc):
        self.typevars[dest_var].union(self.typevars[src_var], loc=loc)

    def lock_type(self, var, tp, loc, literal_value=NOTSET):
        tv = self.typevars[var]
        tv.lock(tp, loc=loc, literal_value=literal_value)

    def propagate_refined_type(self, updated_var, updated_type):
        source_constraint = self.refine_map.get(updated_var)
        if source_constraint is not None:
            source_constraint.refine(self, updated_type)

    def unify(self, raise_errors=True):
        """
        Run the final unification pass over all inferred types, and
        catch imprecise types.
        """
        typdict = utils.UniqueDict()

        def find_offender(name, exhaustive=False):
            offender = None
            for block in self.func_ir.blocks.values():
                offender = block.find_variable_assignment(name)
                if offender is not None:
                    if not exhaustive:
                        break
                    try:
                        hasattr(offender.value, 'name')
                        offender_value = offender.value.name
                    except (AttributeError, KeyError):
                        break
                    orig_offender = offender
                    if offender_value.startswith('$'):
                        offender = find_offender(offender_value, exhaustive=exhaustive)
                        if offender is None:
                            offender = orig_offender
                    break
            return offender

        def diagnose_imprecision(offender):
            list_msg = '\n\nFor Numba to be able to compile a list, the list must have a known and\nprecise type that can be inferred from the other variables. Whilst sometimes\nthe type of empty lists can be inferred, this is not always the case, see this\ndocumentation for help:\n\nhttps://numba.readthedocs.io/en/stable/user/troubleshoot.html#my-code-has-an-untyped-list-problem\n'
            if offender is not None:
                if hasattr(offender, 'value'):
                    if hasattr(offender.value, 'op'):
                        if offender.value.op == 'build_list':
                            return list_msg
                        elif offender.value.op == 'call':
                            try:
                                call_name = offender.value.func.name
                                offender = find_offender(call_name)
                                if isinstance(offender.value, ir.Global):
                                    if offender.value.name == 'list':
                                        return list_msg
                            except (AttributeError, KeyError):
                                pass
            return ''

        def check_var(name):
            tv = self.typevars[name]
            if not tv.defined:
                if raise_errors:
                    offender = find_offender(name)
                    val = getattr(offender, 'value', 'unknown operation')
                    loc = getattr(offender, 'loc', ir.unknown_loc)
                    msg = "Type of variable '%s' cannot be determined, operation: %s, location: %s"
                    raise TypingError(msg % (var, val, loc), loc)
                else:
                    typdict[var] = types.unknown
                    return
            tp = tv.getone()
            if isinstance(tp, types.UndefinedFunctionType):
                tp = tp.get_precise()
            if not tp.is_precise():
                offender = find_offender(name, exhaustive=True)
                msg = "Cannot infer the type of variable '%s'%s, have imprecise type: %s. %s"
                istmp = ' (temporary variable)' if var.startswith('$') else ''
                loc = getattr(offender, 'loc', ir.unknown_loc)
                extra_msg = diagnose_imprecision(offender)
                if raise_errors:
                    raise TypingError(msg % (var, istmp, tp, extra_msg), loc)
                else:
                    typdict[var] = types.unknown
                    return
            else:
                typdict[var] = tp
        temps = set((k for k in self.typevars if not k[0].isalpha()))
        others = set(self.typevars) - temps
        for var in sorted(others):
            check_var(var)
        for var in sorted(temps):
            check_var(var)
        try:
            retty = self.get_return_type(typdict)
        except Exception as e:
            if raise_errors:
                raise e
            else:
                retty = None
        else:
            typdict = utils.UniqueDict(typdict, **{v.name: retty for v in self._get_return_vars()})
        try:
            fntys = self.get_function_types(typdict)
        except Exception as e:
            if raise_errors:
                raise e
            else:
                fntys = None
        if self.generator_info:
            retty = self.get_generator_type(typdict, retty, raise_errors=raise_errors)

        def check_undef_var_in_calls():
            for callnode, calltype in self.calltypes.items():
                if calltype is not None:
                    for i, v in enumerate(calltype.args, start=1):
                        if v is types._undef_var:
                            m = f'undefined variable used in call argument #{i}'
                            raise TypingError(m, loc=callnode.loc)
        check_undef_var_in_calls()
        self.debug.unify_finished(typdict, retty, fntys)
        return (typdict, retty, fntys)

    def get_generator_type(self, typdict, retty, raise_errors=True):
        gi = self.generator_info
        arg_types = [None] * len(self.arg_names)
        for index, name in self.arg_names.items():
            arg_types[index] = typdict[name]
        state_types = None
        try:
            state_types = [typdict[var_name] for var_name in gi.state_vars]
        except KeyError:
            msg = 'Cannot type generator: state variable types cannot be found'
            if raise_errors:
                raise TypingError(msg)
            state_types = [types.unknown for _ in gi.state_vars]
        yield_types = None
        try:
            yield_types = [typdict[y.inst.value.name] for y in gi.get_yield_points()]
        except KeyError:
            msg = 'Cannot type generator: yield type cannot be found'
            if raise_errors:
                raise TypingError(msg)
        if not yield_types:
            msg = 'Cannot type generator: it does not yield any value'
            if raise_errors:
                raise TypingError(msg)
            yield_types = [types.unknown for _ in gi.get_yield_points()]
        if not yield_types or all(yield_types) == types.unknown:
            return types.Generator(self.func_id.func, types.unknown, arg_types, state_types, has_finalizer=True)
        yield_type = self.context.unify_types(*yield_types)
        if yield_type is None or isinstance(yield_type, types.Optional):
            msg = 'Cannot type generator: cannot unify yielded types %s'
            yp_highlights = []
            for y in gi.get_yield_points():
                msg = _termcolor.errmsg("Yield of: IR '%s', type '%s', location: %s")
                yp_highlights.append(msg % (str(y.inst), typdict[y.inst.value.name], y.inst.loc.strformat()))
            explain_ty = set()
            for ty in yield_types:
                if isinstance(ty, types.Optional):
                    explain_ty.add(ty.type)
                    explain_ty.add(types.NoneType('none'))
                else:
                    explain_ty.add(ty)
            if raise_errors:
                raise TypingError("Can't unify yield type from the following types: %s" % ', '.join(sorted(map(str, explain_ty))) + '\n\n' + '\n'.join(yp_highlights))
        return types.Generator(self.func_id.func, yield_type, arg_types, state_types, has_finalizer=True)

    def get_function_types(self, typemap):
        """
        Fill and return the calltypes map.
        """
        calltypes = self.calltypes
        for call, constraint in self.calls:
            calltypes[call] = constraint.get_call_signature()
        return calltypes

    def _unify_return_types(self, rettypes):
        if rettypes:
            unified = self.context.unify_types(*rettypes)
            if isinstance(unified, types.FunctionType):
                return unified
            if unified is None or not unified.is_precise():

                def check_type(atype):
                    lst = []
                    for k, v in self.typevars.items():
                        if atype == v.type:
                            lst.append(k)
                    returns = {}
                    for x in reversed(lst):
                        for block in self.func_ir.blocks.values():
                            for instr in block.find_insts(ir.Return):
                                value = instr.value
                                if isinstance(value, ir.Var):
                                    name = value.name
                                else:
                                    pass
                                if x == name:
                                    returns[x] = instr
                                    break
                    interped = ''
                    for name, offender in returns.items():
                        loc = getattr(offender, 'loc', ir.unknown_loc)
                        msg = "Return of: IR name '%s', type '%s', location: %s"
                        interped = msg % (name, atype, loc.strformat())
                    return interped
                problem_str = []
                for xtype in rettypes:
                    problem_str.append(_termcolor.errmsg(check_type(xtype)))
                raise TypingError("Can't unify return type from the following types: %s" % ', '.join(sorted(map(str, rettypes))) + '\n' + '\n'.join(problem_str))
            return unified
        else:
            return types.none

    def get_return_type(self, typemap):
        rettypes = set()
        for var in self._get_return_vars():
            rettypes.add(typemap[var.name])
        retty = self._unify_return_types(rettypes)
        if retty is types._undef_var:
            raise TypingError('return value is undefined')
        return retty

    def get_state_token(self):
        """The algorithm is monotonic.  It can only grow or "refine" the
        typevar map.
        """
        return [tv.type for name, tv in sorted(self.typevars.items())]

    def constrain_statement(self, inst):
        if isinstance(inst, ir.Assign):
            self.typeof_assign(inst)
        elif isinstance(inst, ir.SetItem):
            self.typeof_setitem(inst)
        elif isinstance(inst, ir.StaticSetItem):
            self.typeof_static_setitem(inst)
        elif isinstance(inst, ir.DelItem):
            self.typeof_delitem(inst)
        elif isinstance(inst, ir.SetAttr):
            self.typeof_setattr(inst)
        elif isinstance(inst, ir.Print):
            self.typeof_print(inst)
        elif isinstance(inst, ir.StoreMap):
            self.typeof_storemap(inst)
        elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return, ir.Del)):
            pass
        elif isinstance(inst, (ir.DynamicRaise, ir.DynamicTryRaise)):
            pass
        elif isinstance(inst, (ir.StaticRaise, ir.StaticTryRaise)):
            pass
        elif isinstance(inst, ir.PopBlock):
            pass
        elif type(inst) in typeinfer_extensions:
            f = typeinfer_extensions[type(inst)]
            f(inst, self)
        else:
            msg = 'Unsupported constraint encountered: %s' % inst
            raise UnsupportedError(msg, loc=inst.loc)

    def typeof_setitem(self, inst):
        constraint = SetItemConstraint(target=inst.target, index=inst.index, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_storemap(self, inst):
        constraint = SetItemConstraint(target=inst.dct, index=inst.key, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_static_setitem(self, inst):
        constraint = StaticSetItemConstraint(target=inst.target, index=inst.index, index_var=inst.index_var, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_delitem(self, inst):
        constraint = DelItemConstraint(target=inst.target, index=inst.index, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_setattr(self, inst):
        constraint = SetAttrConstraint(target=inst.target, attr=inst.attr, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_print(self, inst):
        constraint = PrintConstraint(args=inst.args, vararg=inst.vararg, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_assign(self, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            self.typeof_const(inst, inst.target, value.value)
        elif isinstance(value, ir.Var):
            self.constraints.append(Propagate(dst=inst.target.name, src=value.name, loc=inst.loc))
        elif isinstance(value, (ir.Global, ir.FreeVar)):
            self.typeof_global(inst, inst.target, value)
        elif isinstance(value, ir.Arg):
            self.typeof_arg(inst, inst.target, value)
        elif isinstance(value, ir.Expr):
            self.typeof_expr(inst, inst.target, value)
        elif isinstance(value, ir.Yield):
            self.typeof_yield(inst, inst.target, value)
        else:
            msg = 'Unsupported assignment encountered: %s %s' % (type(value), str(value))
            raise UnsupportedError(msg, loc=inst.loc)

    def resolve_value_type(self, inst, val):
        """
        Resolve the type of a simple Python value, such as can be
        represented by literals.
        """
        try:
            return self.context.resolve_value_type(val)
        except ValueError as e:
            msg = str(e)
        raise TypingError(msg, loc=inst.loc)

    def typeof_arg(self, inst, target, arg):
        src_name = self._mangle_arg_name(arg.name)
        self.constraints.append(ArgConstraint(dst=target.name, src=src_name, loc=inst.loc))

    def typeof_const(self, inst, target, const):
        ty = self.resolve_value_type(inst, const)
        if inst.value.use_literal_type:
            lit = types.maybe_literal(value=const)
        else:
            lit = None
        self.add_type(target.name, lit or ty, loc=inst.loc)

    def typeof_yield(self, inst, target, yield_):
        self.add_type(target.name, types.none, loc=inst.loc)

    def sentry_modified_builtin(self, inst, gvar):
        """
        Ensure that builtins are not modified.
        """
        if gvar.name == 'range' and gvar.value is not range:
            bad = True
        elif gvar.name == 'slice' and gvar.value is not slice:
            bad = True
        elif gvar.name == 'len' and gvar.value is not len:
            bad = True
        else:
            bad = False
        if bad:
            raise TypingError("Modified builtin '%s'" % gvar.name, loc=inst.loc)

    def resolve_call(self, fnty, pos_args, kw_args):
        """
        Resolve a call to a given function type.  A signature is returned.
        """
        if isinstance(fnty, types.FunctionType):
            return fnty.get_call_type(self, pos_args, kw_args)
        if isinstance(fnty, types.RecursiveCall) and (not self._skip_recursion):
            disp = fnty.dispatcher_type.dispatcher
            pysig, args = disp.fold_argument_types(pos_args, kw_args)
            frame = self.context.callstack.match(disp.py_func, args)
            if frame is None:
                sig = self.context.resolve_function_type(fnty.dispatcher_type, pos_args, kw_args)
                fndesc = disp.overloads[args].fndesc
                qual = qualifying_prefix(fndesc.modname, fndesc.qualname)
                fnty.add_overloads(args, qual, fndesc.uid)
                return sig
            fnid = frame.func_id
            qual = qualifying_prefix(fnid.modname, fnid.func_qualname)
            fnty.add_overloads(args, qual, fnid.unique_id)
            return_type = frame.typeinfer.return_types_from_partial()
            if return_type is None:
                raise TypingError('cannot type infer runaway recursion')
            sig = typing.signature(return_type, *args)
            sig = sig.replace(pysig=pysig)
            frame.add_return_type(return_type)
            return sig
        else:
            return self.context.resolve_function_type(fnty, pos_args, kw_args)

    def typeof_global(self, inst, target, gvar):
        try:
            typ = self.resolve_value_type(inst, gvar.value)
        except TypingError as e:
            if gvar.name == self.func_id.func_name and gvar.name in _temporary_dispatcher_map:
                typ = types.Dispatcher(_temporary_dispatcher_map[gvar.name])
            else:
                from numba.misc import special
                nm = gvar.name
                func_glbls = self.func_id.func.__globals__
                if nm not in func_glbls.keys() and nm not in special.__all__ and (nm not in __builtins__.keys()) and (nm not in self.func_id.code.co_freevars):
                    errstr = "NameError: name '%s' is not defined"
                    msg = _termcolor.errmsg(errstr % nm)
                    e.patch_message(msg)
                    raise
                else:
                    msg = _termcolor.errmsg("Untyped global name '%s':" % nm)
                msg += ' %s'
                if nm in special.__all__:
                    tmp = "\n'%s' looks like a Numba internal function, has it been imported (i.e. 'from numba import %s')?\n" % (nm, nm)
                    msg += _termcolor.errmsg(tmp)
                e.patch_message(msg % e)
                raise
        if isinstance(typ, types.Dispatcher) and typ.dispatcher.is_compiling:
            callstack = self.context.callstack
            callframe = callstack.findfirst(typ.dispatcher.py_func)
            if callframe is not None:
                typ = types.RecursiveCall(typ)
            else:
                raise NotImplementedError('call to %s: unsupported recursion' % typ.dispatcher)
        if isinstance(typ, types.Array):
            typ = typ.copy(readonly=True)
        if isinstance(typ, types.BaseAnonymousTuple):
            literaled = [types.maybe_literal(x) for x in gvar.value]
            if all(literaled):
                typ = types.Tuple(literaled)

            def mark_array_ro(tup):
                newtup = []
                for item in tup.types:
                    if isinstance(item, types.Array):
                        item = item.copy(readonly=True)
                    elif isinstance(item, types.BaseAnonymousTuple):
                        item = mark_array_ro(item)
                    newtup.append(item)
                return types.BaseTuple.from_types(newtup)
            typ = mark_array_ro(typ)
        self.sentry_modified_builtin(inst, gvar)
        lit = types.maybe_literal(gvar.value)
        tv = self.typevars[target.name]
        if tv.locked:
            tv.add_type(lit or typ, loc=inst.loc)
        else:
            self.lock_type(target.name, lit or typ, loc=inst.loc)
        self.assumed_immutables.add(inst)

    def typeof_expr(self, inst, target, expr):
        if expr.op == 'call':
            self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext'):
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value)
        elif expr.op == 'exhaust_iter':
            constraint = ExhaustIterConstraint(target.name, count=expr.count, iterator=expr.value, loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'pair_first':
            constraint = PairFirstConstraint(target.name, pair=expr.value, loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'pair_second':
            constraint = PairSecondConstraint(target.name, pair=expr.value, loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'inplace_binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'unary':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.value)
        elif expr.op == 'static_getitem':
            constraint = StaticGetItemConstraint(target.name, value=expr.value, index=expr.index, index_var=expr.index_var, loc=expr.loc)
            self.constraints.append(constraint)
            self.calls.append((inst.value, constraint))
        elif expr.op == 'getitem':
            self.typeof_intrinsic_call(inst, target, operator.getitem, expr.value, expr.index)
        elif expr.op == 'typed_getitem':
            constraint = TypedGetItemConstraint(target.name, value=expr.value, dtype=expr.dtype, index=expr.index, loc=expr.loc)
            self.constraints.append(constraint)
            self.calls.append((inst.value, constraint))
        elif expr.op == 'getattr':
            constraint = GetAttrConstraint(target.name, attr=expr.attr, value=expr.value, loc=inst.loc, inst=inst)
            self.constraints.append(constraint)
        elif expr.op == 'build_tuple':
            constraint = BuildTupleConstraint(target.name, items=expr.items, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_list':
            constraint = BuildListConstraint(target.name, items=expr.items, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_set':
            constraint = BuildSetConstraint(target.name, items=expr.items, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_map':
            constraint = BuildMapConstraint(target.name, items=expr.items, special_value=expr.literal_value, value_indexes=expr.value_indexes, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'cast':
            self.constraints.append(Propagate(dst=target.name, src=expr.value.name, loc=inst.loc))
        elif expr.op == 'phi':
            for iv in expr.incoming_values:
                if iv is not ir.UNDEFINED:
                    self.constraints.append(Propagate(dst=target.name, src=iv.name, loc=inst.loc))
        elif expr.op == 'make_function':
            self.lock_type(target.name, types.MakeFunctionLiteral(expr), loc=inst.loc, literal_value=expr)
        elif expr.op == 'undef':
            self.add_type(target.name, types._undef_var, loc=inst.loc)
        else:
            msg = 'Unsupported op-code encountered: %s' % expr
            raise UnsupportedError(msg, loc=inst.loc)

    def typeof_call(self, inst, target, call):
        constraint = CallConstraint(target.name, call.func.name, call.args, call.kws, call.vararg, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst.value, constraint))

    def typeof_intrinsic_call(self, inst, target, func, *args):
        constraint = IntrinsicCallConstraint(target.name, func, args, kws=(), vararg=None, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst.value, constraint))