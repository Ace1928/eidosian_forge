from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
def _update_removing(target: Any, changes: Any) -> None:
    """Like dict.update(), but remove keys where the value is None."""
    for k, v in changes.items():
        if v is None:
            target.pop(k, None)
        else:
            target[k] = v