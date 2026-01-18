import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Any, Callable
from aiohttp.web import Response
import ray
import ray.dashboard.consts as dashboard_consts
from ray._private.ray_constants import RAY_INTERNAL_DASHBOARD_NAMESPACE, env_bool
from ray.dashboard.optional_deps import PathLike, RouteDef, aiohttp, hdrs
from ray.dashboard.utils import CustomEncoder, to_google_style
def _update_cache(task):
    try:
        response = task.result()
    except Exception:
        response = rest_response(success=False, message=traceback.format_exc())
    data = {'status': response.status, 'headers': dict(response.headers), 'body': response.body}
    cache[key] = _AiohttpCacheValue(data, time.time() + ttl_seconds, task)
    cache.move_to_end(key)
    if len(cache) > maxsize:
        cache.popitem(last=False)
    return response