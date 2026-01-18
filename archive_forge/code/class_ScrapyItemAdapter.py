import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class ScrapyItemAdapter(_MixinDictScrapyItemAdapter, AdapterInterface):

    @classmethod
    def is_item(cls, item: Any) -> bool:
        return isinstance(item, _scrapy_item_classes)

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return issubclass(item_class, _scrapy_item_classes)

    @classmethod
    def get_field_meta_from_class(cls, item_class: type, field_name: str) -> MappingProxyType:
        return MappingProxyType(item_class.fields[field_name])

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
        return list(item_class.fields.keys())

    def field_names(self) -> KeysView:
        return KeysView(self.item.fields)