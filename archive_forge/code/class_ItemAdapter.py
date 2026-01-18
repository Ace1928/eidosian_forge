import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class ItemAdapter(MutableMapping):
    """Wrapper class to interact with data container objects. It provides a common interface
    to extract and set data without having to take the object's type into account.
    """
    ADAPTER_CLASSES: Deque[Type[AdapterInterface]] = deque([ScrapyItemAdapter, DictAdapter, DataclassAdapter, AttrsAdapter, PydanticAdapter])

    def __init__(self, item: Any) -> None:
        for cls in self.ADAPTER_CLASSES:
            if cls.is_item(item):
                self.adapter = cls(item)
                break
        else:
            raise TypeError(f'No adapter found for objects of type: {type(item)} ({item})')

    @classmethod
    def is_item(cls, item: Any) -> bool:
        for adapter_class in cls.ADAPTER_CLASSES:
            if adapter_class.is_item(item):
                return True
        return False

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        for adapter_class in cls.ADAPTER_CLASSES:
            if adapter_class.is_item_class(item_class):
                return True
        return False

    @classmethod
    def _get_adapter_class(cls, item_class: type) -> Type[AdapterInterface]:
        for adapter_class in cls.ADAPTER_CLASSES:
            if adapter_class.is_item_class(item_class):
                return adapter_class
        raise TypeError(f'{item_class} is not a valid item class')

    @classmethod
    def get_field_meta_from_class(cls, item_class: type, field_name: str) -> MappingProxyType:
        adapter_class = cls._get_adapter_class(item_class)
        return adapter_class.get_field_meta_from_class(item_class, field_name)

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
        adapter_class = cls._get_adapter_class(item_class)
        return adapter_class.get_field_names_from_class(item_class)

    @property
    def item(self) -> Any:
        return self.adapter.item

    def __repr__(self) -> str:
        values = ', '.join([f'{key}={value!r}' for key, value in self.items()])
        return f'<{self.__class__.__name__} for {self.item.__class__.__name__}({values})>'

    def __getitem__(self, field_name: str) -> Any:
        return self.adapter.__getitem__(field_name)

    def __setitem__(self, field_name: str, value: Any) -> None:
        self.adapter.__setitem__(field_name, value)

    def __delitem__(self, field_name: str) -> None:
        self.adapter.__delitem__(field_name)

    def __iter__(self) -> Iterator:
        return self.adapter.__iter__()

    def __len__(self) -> int:
        return self.adapter.__len__()

    def get_field_meta(self, field_name: str) -> MappingProxyType:
        """Return metadata for the given field name."""
        return self.adapter.get_field_meta(field_name)

    def field_names(self) -> KeysView:
        """Return read-only key view with the names of all the defined fields for the item."""
        return self.adapter.field_names()

    def asdict(self) -> dict:
        """Return a dict object with the contents of the adapter. This works slightly different
        than calling `dict(adapter)`: it's applied recursively to nested items (if there are any).
        """
        return {key: self._asdict(value) for key, value in self.items()}

    @classmethod
    def _asdict(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: cls._asdict(value) for key, value in obj.items()}
        if isinstance(obj, (list, set, tuple)):
            return obj.__class__((cls._asdict(x) for x in obj))
        if isinstance(obj, cls):
            return obj.asdict()
        if cls.is_item(obj):
            return cls(obj).asdict()
        return obj