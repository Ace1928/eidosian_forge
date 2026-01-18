import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
class Serialize:
    """Safe-ish serialization interface that doesn't rely on Pickle

    Attributes:
        __serialize_fields__ (List[str]): Fields (aka attributes) to serialize.
        __serialize_namespace__ (list): List of classes that deserialization is allowed to instantiate.
                                        Should include all field types that aren't builtin types.
    """

    def memo_serialize(self, types_to_memoize: List) -> Any:
        memo = SerializeMemoizer(types_to_memoize)
        return (self.serialize(memo), memo.serialize())

    def serialize(self, memo=None) -> Dict[str, Any]:
        if memo and memo.in_types(self):
            return {'@': memo.memoized.get(self)}
        fields = getattr(self, '__serialize_fields__')
        res = {f: _serialize(getattr(self, f), memo) for f in fields}
        res['__type__'] = type(self).__name__
        if hasattr(self, '_serialize'):
            self._serialize(res, memo)
        return res

    @classmethod
    def deserialize(cls: Type[_T], data: Dict[str, Any], memo: Dict[int, Any]) -> _T:
        namespace = getattr(cls, '__serialize_namespace__', [])
        namespace = {c.__name__: c for c in namespace}
        fields = getattr(cls, '__serialize_fields__')
        if '@' in data:
            return memo[data['@']]
        inst = cls.__new__(cls)
        for f in fields:
            try:
                setattr(inst, f, _deserialize(data[f], namespace, memo))
            except KeyError as e:
                raise KeyError('Cannot find key for class', cls, e)
        if hasattr(inst, '_deserialize'):
            inst._deserialize()
        return inst