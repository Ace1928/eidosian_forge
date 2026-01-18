from __future__ import annotations
import contextlib
import collections.abc
from threading import Lock
from asyncio import Lock as AsyncLock
from pathlib import Path
from pydantic import BaseModel
from typing import TypeVar, Generic, Any, Dict, Optional, Union, Iterable, List, Type, ItemsView, TYPE_CHECKING
from lazyops.utils.helpers import  create_unique_id
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from ..serializers import get_serializer, SerializerT
from ..serializers.base import create_obj_hash
def _precheck(self, **kwargs):
    """
        Run a precheck operation
        """
    pass