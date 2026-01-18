import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
class Data(Expr):

    def __init__(self, code: str, ctype: _cuda_types.TypeBase) -> None:
        assert isinstance(code, str)
        assert isinstance(ctype, _cuda_types.TypeBase)
        self.code = code
        self.ctype = ctype
        if not isinstance(ctype, _cuda_types.Unknown):
            try:
                self.__doc__ = f'{str(ctype)} {code}\n{ctype.__doc__}'
            except NotImplementedError:
                self.__doc__ = f'{code}'

    @property
    def obj(self):
        raise ValueError(f'Constant value is requried: {self.code}')

    def __repr__(self) -> str:
        return f'<Data code = "{self.code}", type = {self.ctype}>'

    @classmethod
    def init(cls, x: Expr, env) -> 'Data':
        if isinstance(x, Data):
            return x
        if isinstance(x, Constant):
            if isinstance(x.obj, tuple):
                elts = [Data.init(Constant(e), env) for e in x.obj]
                elts_code = ', '.join([e.code for e in elts])
                if len(elts) == 2:
                    return Data(f'thrust::make_pair({elts_code})', _cuda_types.Tuple([x.ctype for x in elts]))
                return Data(f'thrust::make_tuple({elts_code})', _cuda_types.Tuple([x.ctype for x in elts]))
            ctype = _cuda_typerules.get_ctype_from_scalar(env.mode, x.obj)
            code = _cuda_types.get_cuda_code_from_constant(x.obj, ctype)
            return Data(code, ctype)
        raise TypeError(f"'{x}' cannot be interpreted as a cuda object.")