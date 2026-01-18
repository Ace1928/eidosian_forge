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
class OAuthAuthenticator(BaseSaslAuthenticator):

    def __init__(self, *, sasl_oauth_token_provider):
        self._sasl_oauth_token_provider = sasl_oauth_token_provider
        self._token_sent = False

    async def step(self, payload):
        if self._token_sent:
            return
        token = await self._sasl_oauth_token_provider.token()
        token_extensions = self._token_extensions()
        self._token_sent = True
        return (self._build_oauth_client_request(token, token_extensions).encode('utf-8'), True)

    def _build_oauth_client_request(self, token, token_extensions):
        return f'n,,\x01auth=Bearer {token}{token_extensions}\x01\x01'

    def _token_extensions(self):
        """
        Return a string representation of the OPTIONAL key-value pairs
        that can be sent with an OAUTHBEARER initial request.
        """
        if callable(getattr(self._sasl_oauth_token_provider, 'extensions', None)):
            extensions = self._sasl_oauth_token_provider.extensions()
            if len(extensions) > 0:
                msg = '\x01'.join([f'{k}={v}' for k, v in extensions.items()])
                return '\x01' + msg
        return ''