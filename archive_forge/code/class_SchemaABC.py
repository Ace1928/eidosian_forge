from __future__ import annotations
from abc import ABC, abstractmethod
class SchemaABC(ABC):
    """Abstract base class from which all Schemas inherit."""

    @abstractmethod
    def dump(self, obj, *, many: bool | None=None):
        pass

    @abstractmethod
    def dumps(self, obj, *, many: bool | None=None):
        pass

    @abstractmethod
    def load(self, data, *, many: bool | None=None, partial=None, unknown=None):
        pass

    @abstractmethod
    def loads(self, json_data, *, many: bool | None=None, partial=None, unknown=None, **kwargs):
        pass