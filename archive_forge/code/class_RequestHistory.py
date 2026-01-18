from __future__ import annotations
import email
import logging
import random
import re
import time
import typing
from itertools import takewhile
from types import TracebackType
from ..exceptions import (
from .util import reraise
class RequestHistory(typing.NamedTuple):
    method: str | None
    url: str | None
    error: Exception | None
    status: int | None
    redirect_location: str | None