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
def _next_available_name(self) -> str | None:
    for n in itertools.count(start=1):
        name = self.name_template % n
        if name not in self.terminals:
            return name
    return None