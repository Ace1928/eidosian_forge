import asyncio
import inspect
import json
import logging
import warnings
import zlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from redis import WatchError
from .defaults import CALLBACK_TIMEOUT, UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .timeouts import BaseDeathPenalty, JobTimeoutException
from .connections import resolve_connection
from .exceptions import DeserializationError, InvalidJobOperation, NoSuchJobError
from .local import LocalStack
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import (
def _handle_success(self, result_ttl: int, pipeline: 'Pipeline'):
    """Saves and cleanup job after successful execution"""
    self.set_status(JobStatus.FINISHED, pipeline=pipeline)
    include_result = not self.supports_redis_streams
    self.save(pipeline=pipeline, include_meta=False, include_result=include_result)
    if self.supports_redis_streams:
        from .results import Result
        Result.create(self, Result.Type.SUCCESSFUL, return_value=self._result, ttl=result_ttl, pipeline=pipeline)
    if result_ttl != 0:
        finished_job_registry = self.finished_job_registry
        finished_job_registry.add(self, result_ttl, pipeline)