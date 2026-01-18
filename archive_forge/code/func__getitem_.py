import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
@_tp_cache
def _getitem_(self, params):
    if params is None:
        raise TypeError(f'{self.__name__}[t]: t can not be None')
    if isinstance(params, str):
        params = ForwardRef(params)
    if not isinstance(params, tuple):
        params = (params,)
    msg = f'{self.__name__}[t]: t must be a type'
    params = tuple((_type_check(p, msg) for p in params))
    if isinstance(self.type.param, _GenericAlias):
        orig = getattr(self.type.param, '__origin__', None)
        if isinstance(orig, type) and orig is not Generic:
            p = self.type.param[params]
            t = _DataPipeType(p)
            l = len(str(self.type)) + 2
            name = self.__name__[:-l]
            name = name + '[' + str(t) + ']'
            bases = (self,) + self.__bases__
            return self.__class__(name, bases, {'__init_subclass__': _dp_init_subclass, 'type': t, '__type_class__': True})
    if len(params) > 1:
        raise TypeError(f'Too many parameters for {self} actual {len(params)}, expected 1')
    t = _DataPipeType(params[0])
    if not t.issubtype(self.type):
        raise TypeError(f'Can not subclass a DataPipe[{t}] from DataPipe[{self.type}]')
    if self.type == t:
        return self
    name = self.__name__ + '[' + str(t) + ']'
    bases = (self,) + self.__bases__
    return self.__class__(name, bases, {'__init_subclass__': _dp_init_subclass, '__type_class__': True, 'type': t})