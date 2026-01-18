import asyncio
import logging
import os
import random
from asyncio.events import AbstractEventLoop
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, DefaultDict, Dict, Optional, Set, Tuple, Union
import ray
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.common import ReplicaName
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import format_actor_name
from ray.serve.generated.serve_pb2 import ActorNameList
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet, LongPollRequest, LongPollResult
from ray.serve.generated.serve_pb2 import UpdatedObject as UpdatedObjectProto
from ray.util import metrics
def _parse_xlang_key(self, xlang_key: str) -> KeyType:
    if xlang_key is None:
        raise ValueError('func _parse_xlang_key: xlang_key is None')
    if xlang_key.startswith('(') and xlang_key.endswith(')'):
        fields = xlang_key[1:-1].split(',')
        if len(fields) == 2:
            enum_field = self._parse_poll_namespace(fields[0].strip())
            if isinstance(enum_field, LongPollNamespace):
                return (enum_field, fields[1].strip())
    else:
        return self._parse_poll_namespace(xlang_key)
    raise ValueError('can not parse key type from xlang_key {}'.format(xlang_key))