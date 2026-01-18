from __future__ import annotations
import logging # isort:skip
import asyncio
from functools import wraps
from typing import (
class NoLockCallback(Protocol[F]):
    __call__: F
    nolock: Literal[True]