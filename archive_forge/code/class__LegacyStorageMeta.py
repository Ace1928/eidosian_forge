import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
class _LegacyStorageMeta(type):
    dtype: torch.dtype

    def __instancecheck__(cls, instance):
        if type(instance) == TypedStorage:
            cls_device = _get_device_from_module(cls.__module__)
            return cls_device == instance.device.type and cls.dtype == instance.dtype
        return False