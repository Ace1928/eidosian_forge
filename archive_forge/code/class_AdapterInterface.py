import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class AdapterInterface(MutableMapping, metaclass=ABCMeta):
    """Abstract Base Class for adapters.

    An adapter that handles a specific type of item should inherit from this
    class and implement the abstract methods defined here, plus the
    abtract methods inherited from the MutableMapping base class.
    """

    def __init__(self, item: Any) -> None:
        self.item = item

    @classmethod
    @abstractmethod
    def is_item_class(cls, item_class: type) -> bool:
        """Return True if the adapter can handle the given item class, False otherwise."""
        raise NotImplementedError()

    @classmethod
    def is_item(cls, item: Any) -> bool:
        """Return True if the adapter can handle the given item, False otherwise."""
        return cls.is_item_class(item.__class__)

    @classmethod
    def get_field_meta_from_class(cls, item_class: type, field_name: str) -> MappingProxyType:
        return MappingProxyType({})

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
        """Return a list of fields defined for ``item_class``.
        If a class doesn't support fields, None is returned."""
        return None

    def get_field_meta(self, field_name: str) -> MappingProxyType:
        """Return metadata for the given field name, if available."""
        return self.get_field_meta_from_class(self.item.__class__, field_name)

    def field_names(self) -> KeysView:
        """Return a dynamic view of the item's field names."""
        return self.keys()