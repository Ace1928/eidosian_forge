from __future__ import annotations
import os
import json
import socket
import contextlib
import logging
from typing import Optional, Dict, Any, Union, Type, Mapping, Callable, List
from lazyops.utils.logs import default_logger as logger
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.types import BaseSettings, validator, root_validator, lazyproperty, KeyDBUri
from aiokeydb.v2.serializers import SerializerType, BaseSerializer
from aiokeydb.v2.utils import import_string
from aiokeydb.v2.configs.worker import KeyDBWorkerSettings
from aiokeydb.v2.backoff import default_backoff
@lazyproperty
def _retry_exceptions(self) -> List[Type[Exception]]:
    """
        Returns the list of retry exceptions
        """
    _retries = []
    if self.retry_on_timeout:
        self.retry_on_timeout = False
        _retries.extend([TimeoutError, exceptions.TimeoutError])
    if self.retry_on_connection_error:
        _retries.extend([ConnectionError, exceptions.ConnectionError])
    if self.retry_on_connection_reset_error:
        _retries.append(ConnectionResetError)
    if self.retry_on_response_error:
        _retries.extend([exceptions.ResponseError, exceptions.BusyLoadingError])
    return list(set(_retries))