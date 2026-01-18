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
class SaslPlainAuthenticator(BaseSaslAuthenticator):

    def __init__(self, *, loop, sasl_plain_password, sasl_plain_username):
        self._loop = loop
        self._sasl_plain_username = sasl_plain_username
        self._sasl_plain_password = sasl_plain_password
        self._authenticator = self.authenticator_plain()

    def authenticator_plain(self):
        """ Automaton to authenticate with SASL tokens
        """
        data = '\x00'.join([self._sasl_plain_username, self._sasl_plain_username, self._sasl_plain_password]).encode('utf-8')
        resp = (yield (data, True))
        assert resp == b'', 'Server should either close or send an empty response'