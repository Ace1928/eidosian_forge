import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
@staticmethod
def _idle_check(self_ref):
    self = self_ref()
    if self is None:
        return
    idle_for = time.monotonic() - self._last_action
    timeout = self._max_idle_ms / 1000
    if idle_for >= timeout and (not self._requests):
        self.close(CloseReason.IDLE_DROP)
    else:
        if self._requests:
            wake_up_in = timeout
        else:
            wake_up_in = timeout - idle_for
        self._idle_handle = self._loop.call_later(wake_up_in, self._idle_check, self_ref)