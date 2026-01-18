import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
class BuiltinFunc(Expr):

    def call(self, env: 'Environment', *args, **kwargs) -> Expr:
        for x in itertools.chain(args, kwargs.values()):
            if not isinstance(x, Constant):
                raise TypeError('Arguments must be constants.')
        args = tuple([x.obj for x in args])
        kwargs = dict([(k, v.obj) for k, v in kwargs.items()])
        return self.call_const(env, *args, **kwargs)

    def call_const(self, env: 'Environment', *args: Any, **kwarg: Any) -> Expr:
        raise NotImplementedError

    def __init__(self) -> None:
        self.__doc__ = type(self).__call__.__doc__

    def __call__(self) -> NoReturn:
        raise RuntimeError('Cannot call this function from Python layer.')

    def __repr__(self) -> str:
        return '<cupyx.jit function>'

    @classmethod
    def from_class_method(cls, method, ctype_self, instance):

        class _Wrapper(BuiltinFunc):

            def call(self, env, *args, **kwargs):
                return method(ctype_self, env, instance, *args)
        return _Wrapper()