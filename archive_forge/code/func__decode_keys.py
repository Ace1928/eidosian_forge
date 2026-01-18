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
def _decode_keys(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _decode_keys(v)
        if isinstance(v, list):
            new_list = []
            for i in v:
                if isinstance(i, dict):
                    new_list.append(_decode_keys(i))
                else:
                    new_list.append(i)
            d[k] = new_list
        elif k in decode_keys:
            d[k] = binary_to_hex(b64decode(v))
        else:
            d[k] = v
    return d