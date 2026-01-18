import inspect
import typing as py_typing
from numba.core.typing.typeof import typeof
from numba.core import errors, types
def _builtin_infer(self, py_type):
    if not isinstance(py_type, py_typing._GenericAlias):
        return
    if getattr(py_type, '__origin__', None) is py_typing.Union:
        if len(py_type.__args__) != 2:
            raise errors.TypingError('Cannot type Union of more than two types')
        arg_1_py, arg_2_py = py_type.__args__
        if arg_2_py is type(None):
            return types.Optional(self.infer(arg_1_py))
        elif arg_1_py is type(None):
            return types.Optional(self.infer(arg_2_py))
        else:
            raise errors.TypingError(f'Cannot type Union that is not an Optional (neither type type {arg_2_py} is not NoneType')
    if getattr(py_type, '__origin__', None) is list:
        element_py, = py_type.__args__
        return types.ListType(self.infer(element_py))
    if getattr(py_type, '__origin__', None) is dict:
        key_py, value_py = py_type.__args__
        return types.DictType(self.infer(key_py), self.infer(value_py))
    if getattr(py_type, '__origin__', None) is set:
        element_py, = py_type.__args__
        return types.Set(self.infer(element_py))
    if getattr(py_type, '__origin__', None) is tuple:
        tys = tuple(map(self.infer, py_type.__args__))
        return types.BaseTuple.from_types(tys)