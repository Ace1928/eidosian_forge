from __future__ import annotations
import datetime
import inspect
import warnings
from collections import OrderedDict, abc
from typing import (
from urllib.parse import unquote_plus
from bson import SON
from bson.binary import UuidRepresentation
from bson.codec_options import CodecOptions, DatetimeConversion, TypeRegistry
from bson.raw_bson import RawBSONDocument
from pymongo.auth import MECHANISMS
from pymongo.compression_support import (
from pymongo.driver_info import DriverInfo
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _validate_event_listeners
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import _MONGOS_MODES, _ServerMode
from pymongo.server_api import ServerApi
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern, validate_boolean
class _CaseInsensitiveDictionary(MutableMapping[str, Any]):

    def __init__(self, *args: Any, **kwargs: Any):
        self.__casedkeys: dict[str, Any] = {}
        self.__data: dict[str, Any] = {}
        self.update(dict(*args, **kwargs))

    def __contains__(self, key: str) -> bool:
        return key.lower() in self.__data

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.__casedkeys)

    def __repr__(self) -> str:
        return str({self.__casedkeys[k]: self.__data[k] for k in self})

    def __setitem__(self, key: str, value: Any) -> None:
        lc_key = key.lower()
        self.__casedkeys[lc_key] = key
        self.__data[lc_key] = value

    def __getitem__(self, key: str) -> Any:
        return self.__data[key.lower()]

    def __delitem__(self, key: str) -> None:
        lc_key = key.lower()
        del self.__casedkeys[lc_key]
        del self.__data[lc_key]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, abc.Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        for key in other:
            if self[key] != other[key]:
                return False
        return True

    def get(self, key: str, default: Optional[Any]=None) -> Any:
        return self.__data.get(key.lower(), default)

    def pop(self, key: str, *args: Any, **kwargs: Any) -> Any:
        lc_key = key.lower()
        self.__casedkeys.pop(lc_key, None)
        return self.__data.pop(lc_key, *args, **kwargs)

    def popitem(self) -> tuple[str, Any]:
        lc_key, cased_key = self.__casedkeys.popitem()
        value = self.__data.pop(lc_key)
        return (cased_key, value)

    def clear(self) -> None:
        self.__casedkeys.clear()
        self.__data.clear()

    @overload
    def setdefault(self, key: str, default: None=None) -> Optional[Any]:
        ...

    @overload
    def setdefault(self, key: str, default: Any) -> Any:
        ...

    def setdefault(self, key: str, default: Optional[Any]=None) -> Optional[Any]:
        lc_key = key.lower()
        if key in self:
            return self.__data[lc_key]
        else:
            self.__casedkeys[lc_key] = key
            self.__data[lc_key] = default
            return default

    def update(self, other: Mapping[str, Any]) -> None:
        if isinstance(other, _CaseInsensitiveDictionary):
            for key in other:
                self[other.cased_key(key)] = other[key]
        else:
            for key in other:
                self[key] = other[key]

    def cased_key(self, key: str) -> Any:
        return self.__casedkeys[key.lower()]