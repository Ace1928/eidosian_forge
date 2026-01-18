import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
class NotifyQueue:
    """Asyncio notify queue for Dict signal."""
    _queue = asyncio.Queue()

    @classmethod
    def put(cls, co):
        cls._queue.put_nowait(co)

    @classmethod
    async def get(cls):
        return await cls._queue.get()