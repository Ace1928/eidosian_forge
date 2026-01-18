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
def _send_sasl_token(self, payload, expect_response=True):
    if self._writer is None:
        raise Errors.KafkaConnectionError(f'No connection to broker at {self._host}:{self._port}')
    size = struct.pack('>i', len(payload))
    try:
        self._writer.write(size + payload)
    except OSError as err:
        self.close(reason=CloseReason.CONNECTION_BROKEN)
        raise Errors.KafkaConnectionError(f'Connection at {self._host}:{self._port} broken: {err}')
    if not expect_response:
        return self._writer.drain()
    fut = self._loop.create_future()
    self._requests.append((None, None, fut))
    return wait_for(fut, self._request_timeout)