from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
def _metadata_env() -> dict[str, Any]:
    env: dict[str, Any] = {}
    if (_is_lambda(), _is_azure_func(), _is_gcp_func(), _is_vercel()).count(True) != 1:
        return env
    if _is_lambda():
        env['name'] = 'aws.lambda'
        region = os.getenv('AWS_REGION')
        if region:
            env['region'] = region
        memory_mb = _getenv_int('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
        if memory_mb is not None:
            env['memory_mb'] = memory_mb
    elif _is_azure_func():
        env['name'] = 'azure.func'
    elif _is_gcp_func():
        env['name'] = 'gcp.func'
        region = os.getenv('FUNCTION_REGION')
        if region:
            env['region'] = region
        memory_mb = _getenv_int('FUNCTION_MEMORY_MB')
        if memory_mb is not None:
            env['memory_mb'] = memory_mb
        timeout_sec = _getenv_int('FUNCTION_TIMEOUT_SEC')
        if timeout_sec is not None:
            env['timeout_sec'] = timeout_sec
    elif _is_vercel():
        env['name'] = 'vercel'
        region = os.getenv('VERCEL_REGION')
        if region:
            env['region'] = region
    return env