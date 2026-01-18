from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
class _BaseCodecOptions(NamedTuple):
    document_class: Type[Mapping[str, Any]]
    tz_aware: bool
    uuid_representation: int
    unicode_decode_error_handler: str
    tzinfo: Optional[datetime.tzinfo]
    type_registry: TypeRegistry
    datetime_conversion: Optional[DatetimeConversion]