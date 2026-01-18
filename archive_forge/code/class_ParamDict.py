import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
class ParamDict(IndexedOrderedDict[str, Any]):
    """Parameter dictionary, a subclass of ``IndexedOrderedDict``, keys must be string

    :param data: for possible types, see :func:`~triad.utils.iter.to_kv_iterable`
    :param deep: whether to deep copy ``data``
    """
    OVERWRITE = 0
    THROW = 1
    IGNORE = 2

    def __init__(self, data: Any=None, deep: bool=True):
        super().__init__()
        self.update(data, deep=deep)

    def __setitem__(self, key: str, value: Any, *args: Any, **kwds: Any) -> None:
        assert isinstance(key, str)
        super().__setitem__(key, value, *args, **kwds)

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            key = self.get_key_by_index(key)
        return super().__getitem__(key)

    def get(self, key: Union[int, str], default: Any) -> Any:
        """Get value by ``key``, and the value must be a subtype of the type of
        ``default``(which can't be None). If the ``key`` is not found,
        return ``default``.

        :param key: the key to search
        :raises NoneArgumentError: if default is None
        :raises TypeError: if the value can't be converted to the type of ``default``

        :return: the value by ``key``, and the value must be a subtype of the type of
            ``default``. If ``key`` is not found, return `default`
        """
        assert_arg_not_none(default, 'default')
        if isinstance(key, str) and key in self or isinstance(key, int):
            return as_type(self[key], type(default))
        return default

    def get_or_none(self, key: Union[int, str], expected_type: type) -> Any:
        """Get value by `key`, and the value must be a subtype of ``expected_type``

        :param key: the key to search
        :param expected_type: expected return value type

        :raises TypeError: if the value can't be converted to ``expected_type``

        :return: if ``key`` is not found, None. Otherwise if the value can be converted
            to ``expected_type``, return the converted value, otherwise raise exception
        """
        return self._get_or(key, expected_type, throw=False)

    def get_or_throw(self, key: Union[int, str], expected_type: type) -> Any:
        """Get value by ``key``, and the value must be a subtype of ``expected_type``.
        If ``key`` is not found or value can't be converted to ``expected_type``, raise
        exception

        :param key: the key to search
        :param expected_type: expected return value type

        :raises KeyError: if ``key`` is not found
        :raises TypeError: if the value can't be converted to ``expected_type``

        :return: only when ``key`` is found and can be converted to ``expected_type``,
            return the converted value
        """
        return self._get_or(key, expected_type, throw=True)

    def to_json(self, indent: bool=False) -> str:
        """Generate json expression string for the dictionary

        :param indent: whether to have indent
        :return: json string
        """
        if not indent:
            return json.dumps(self, separators=(',', ':'))
        else:
            return json.dumps(self, indent=4)

    def update(self, other: Any, on_dup: int=0, deep: bool=True) -> 'ParamDict':
        """Update dictionary with another object (for possible types,
        see :func:`~triad.utils.iter.to_kv_iterable`)

        :param other: for possible types, see :func:`~triad.utils.iter.to_kv_iterable`
        :param on_dup: one of ``ParamDict.OVERWRITE``, ``ParamDict.THROW``
            and ``ParamDict.IGNORE``

        :raises KeyError: if using ``ParamDict.THROW`` and other contains existing keys
        :raises ValueError: if ``on_dup`` is invalid
        :return: itself
        """
        self._pre_update('update', True)
        for k, v in to_kv_iterable(other):
            if on_dup == ParamDict.OVERWRITE or k not in self:
                self[k] = copy.deepcopy(v) if deep else v
            elif on_dup == ParamDict.THROW:
                raise KeyError(f'{k} exists in dict')
            elif on_dup == ParamDict.IGNORE:
                continue
            else:
                raise ValueError(f'{on_dup} is not supported')
        return self

    def _get_or(self, key: Union[int, str], expected_type: type, throw: bool=True) -> Any:
        if isinstance(key, str) and key in self or isinstance(key, int):
            return as_type(self[key], expected_type)
        if throw:
            raise KeyError(f'{key} not found')
        return None