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
def _dumps_json_single(obj: Any, default: Optional[Callable[[Any], Any]]=None) -> bytes:
    try:
        return orjson.dumps(obj, default=default, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS | orjson.OPT_SERIALIZE_UUID | orjson.OPT_NON_STR_KEYS)
    except TypeError as e:
        logger.debug(f'Orjson serialization failed: {repr(e)}. Falling back to json.')
        result = json.dumps(obj, default=_simple_default, ensure_ascii=True).encode('utf-8')
        try:
            result = orjson.dumps(orjson.loads(result.decode('utf-8', errors='lossy')))
        except orjson.JSONDecodeError:
            result = _elide_surrogates(result)
        return result