import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
class IndexedOrderedDict(OrderedDict, Dict[KT, VT]):
    """Subclass of OrderedDict that can get and set with index"""

    def __init__(self, *args: Any, **kwds: Any):
        self._readonly = False
        self._need_reindex = True
        self._key_index: Dict[Any, int] = {}
        self._index_key: List[Any] = []
        super().__init__(*args, **kwds)

    @property
    def readonly(self) -> bool:
        """Whether this dict is readonly"""
        return self._readonly

    def set_readonly(self) -> None:
        """Make this dict readonly"""
        self._readonly = True

    def index_of_key(self, key: Any) -> int:
        """Get index of key

        :param key: key value
        :return: index of the key value
        """
        self._build_index()
        return self._key_index[key]

    def get_key_by_index(self, index: int) -> KT:
        """Get key by index

        :param index: index of the key
        :return: key value at the index
        """
        self._build_index()
        return self._index_key[index]

    def get_value_by_index(self, index: int) -> VT:
        """Get value by index

        :param index: index of the item
        :return: value at the index
        """
        key = self.get_key_by_index(index)
        return self[key]

    def get_item_by_index(self, index: int) -> Tuple[KT, VT]:
        """Get key value pair by index

        :param index: index of the item
        :return: key value tuple at the index
        """
        key = self.get_key_by_index(index)
        return (key, self[key])

    def set_value_by_index(self, index: int, value: VT) -> None:
        """Set value by index

        :param index: index of the item
        :param value: new value
        """
        key = self.get_key_by_index(index)
        self[key] = value

    def pop_by_index(self, index: int) -> Tuple[KT, VT]:
        """Pop item at index

        :param index: index of the item
        :return: key value tuple at the index
        """
        key = self.get_key_by_index(index)
        return (key, self.pop(key))

    def equals(self, other: Any, with_order: bool):
        """Compare with another object

        :param other: for possible types, see :func:`~triad.utils.iter.to_kv_iterable`
        :param with_order: whether to compare order
        :return: whether they equal
        """
        if with_order:
            if isinstance(other, OrderedDict):
                return self == other
            return self == OrderedDict(to_kv_iterable(other))
        else:
            if isinstance(other, OrderedDict) or not isinstance(other, Dict):
                return self == dict(to_kv_iterable(other))
            return self == other

    def __setitem__(self, key: KT, value: VT, *args: Any, **kwds: Any) -> None:
        self._pre_update('__setitem__', key not in self)
        super().__setitem__(key, value, *args, **kwds)

    def __delitem__(self, *args: Any, **kwds: Any) -> None:
        self._pre_update('__delitem__')
        super().__delitem__(*args, **kwds)

    def clear(self) -> None:
        self._pre_update('clear')
        super().clear()

    def copy(self) -> 'IndexedOrderedDict':
        other = super().copy()
        assert isinstance(other, IndexedOrderedDict)
        other._need_reindex = self._need_reindex
        other._index_key = self._index_key.copy()
        other._key_index = self._key_index.copy()
        other._readonly = False
        return other

    def __copy__(self) -> 'IndexedOrderedDict':
        return self.copy()

    def __deepcopy__(self, arg: Any) -> 'IndexedOrderedDict':
        it = [(copy.deepcopy(k), copy.deepcopy(v)) for k, v in self.items()]
        return IndexedOrderedDict(it)

    def popitem(self, *args: Any, **kwds: Any) -> Tuple[KT, VT]:
        self._pre_update('popitem')
        return super().popitem(*args, **kwds)

    def move_to_end(self, *args: Any, **kwds: Any) -> None:
        self._pre_update('move_to_end')
        super().move_to_end(*args, **kwds)

    def __sizeof__(self) -> int:
        return super().__sizeof__() + sys.getsizeof(self._need_reindex)

    def pop(self, *args: Any, **kwds: Any) -> VT:
        self._pre_update('pop')
        return super().pop(*args, **kwds)

    def _build_index(self) -> None:
        if self._need_reindex:
            self._index_key = list(self.keys())
            self._key_index = {x: i for i, x in enumerate(self._index_key)}
            self._need_reindex = False

    def _pre_update(self, op: str, need_reindex: bool=True) -> None:
        if self.readonly:
            raise InvalidOperationError('This dict is readonly')
        self._need_reindex = need_reindex