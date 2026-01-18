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
class AIOKafkaProtocol(asyncio.StreamReaderProtocol):

    def __init__(self, closed_fut, *args, loop, **kw):
        self._closed_fut = closed_fut
        super().__init__(*args, loop=loop, **kw)

    def connection_lost(self, exc):
        super().connection_lost(exc)
        if not self._closed_fut.cancelled():
            self._closed_fut.set_result(None)