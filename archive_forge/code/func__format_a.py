import datetime
import functools
import logging
import os
import re
import time as time_mod
from collections import namedtuple
from typing import Any, Callable, Dict, Iterable, List, Tuple  # noqa
from .abc import AbstractAccessLogger
from .web_request import BaseRequest
from .web_response import StreamResponse
@staticmethod
def _format_a(request: BaseRequest, response: StreamResponse, time: float) -> str:
    if request is None:
        return '-'
    ip = request.remote
    return ip if ip is not None else '-'