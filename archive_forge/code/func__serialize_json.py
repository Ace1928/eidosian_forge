from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _serialize_json(obj: Any, depth: int=0, serialize_py: bool=True) -> Any:
    try:
        if depth >= _MAX_DEPTH:
            try:
                return orjson.loads(_dumps_json_single(obj))
            except BaseException:
                return repr(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        if isinstance(obj, (set, tuple)):
            return orjson.loads(_dumps_json_single(list(obj)))
        serialization_methods = [('model_dump_json', True), ('json', True), ('to_json', False), ('model_dump', True), ('dict', False)]
        for attr, exclude_none in serialization_methods:
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                try:
                    method = getattr(obj, attr)
                    json_str = method(exclude_none=exclude_none) if exclude_none else method()
                    if isinstance(json_str, str):
                        return json.loads(json_str)
                    return orjson.loads(_dumps_json(json_str, depth=depth + 1, serialize_py=serialize_py))
                except Exception as e:
                    logger.debug(f'Failed to serialize {type(obj)} to JSON: {e}')
                    pass
        if serialize_py:
            all_attrs = {}
            if hasattr(obj, '__slots__'):
                all_attrs.update({slot: getattr(obj, slot, None) for slot in obj.__slots__})
            if hasattr(obj, '__dict__'):
                all_attrs.update(vars(obj))
            if all_attrs:
                filtered = {k: v if v is not obj else repr(v) for k, v in all_attrs.items()}
                return orjson.loads(_dumps_json(filtered, depth=depth + 1, serialize_py=serialize_py))
        return repr(obj)
    except BaseException as e:
        logger.debug(f'Failed to serialize {type(obj)} to JSON: {e}')
        return repr(obj)