from __future__ import annotations
import contextlib
import datetime
from functools import partial
from functools import wraps
import json
import logging
from numbers import Number
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from decorator import decorate
from . import exception
from .api import BackendArguments
from .api import BackendFormatted
from .api import CachedValue
from .api import CacheMutex
from .api import CacheReturnType
from .api import CantDeserializeException
from .api import KeyType
from .api import MetaDataType
from .api import NO_VALUE
from .api import SerializedReturnType
from .api import Serializer
from .api import ValuePayload
from .backends import _backend_loader
from .backends import register_backend  # noqa
from .proxy import ProxyBackend
from .util import function_key_generator
from .util import function_multi_key_generator
from .util import repr_obj
from .. import Lock
from .. import NeedRegenerationException
from ..util import coerce_string_conf
from ..util import memoized_property
from ..util import NameRegistry
from ..util import PluginLoader
from ..util.typing import Self
def _parse_serialized_from_backend(self, value: SerializedReturnType) -> CacheReturnType:
    if value in (None, NO_VALUE):
        return NO_VALUE
    assert self.deserializer
    byte_value = cast(bytes, value)
    bytes_metadata, _, bytes_payload = byte_value.partition(b'|')
    metadata = json.loads(bytes_metadata)
    try:
        payload = self.deserializer(bytes_payload)
    except CantDeserializeException:
        return NO_VALUE
    else:
        return CachedValue(payload, metadata)