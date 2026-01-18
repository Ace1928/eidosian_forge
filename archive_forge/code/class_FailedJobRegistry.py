import calendar
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union
from rq.serializers import resolve_serializer
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_FAILURE_TTL
from .exceptions import AbandonedJobError, InvalidJobOperation, NoSuchJobError
from .job import Job, JobStatus
from .queue import Queue
from .utils import as_text, backend_class, current_timestamp
class FailedJobRegistry(BaseRegistry):
    """
    Registry of containing failed jobs.
    """
    key_template = 'rq:failed:{0}'

    def cleanup(self, timestamp: Optional[float]=None):
        """Remove expired jobs from registry.

        Removes jobs with an expiry time earlier than timestamp, specified as
        seconds since the Unix epoch. timestamp defaults to call time if
        unspecified.
        """
        score = timestamp if timestamp is not None else current_timestamp()
        self.connection.zremrangebyscore(self.key, 0, score)

    def add(self, job: 'Job', ttl=None, exc_string: str='', pipeline: Optional['Pipeline']=None, _save_exc_to_job: bool=False):
        """
        Adds a job to a registry with expiry time of now + ttl.
        `ttl` defaults to DEFAULT_FAILURE_TTL if not specified.
        """
        if ttl is None:
            ttl = DEFAULT_FAILURE_TTL
        score = ttl if ttl < 0 else current_timestamp() + ttl
        if pipeline:
            p = pipeline
        else:
            p = self.connection.pipeline()
        job._exc_info = exc_string
        job.save(pipeline=p, include_meta=False, include_result=_save_exc_to_job)
        job.cleanup(ttl=ttl, pipeline=p)
        p.zadd(self.key, {job.id: score})
        if not pipeline:
            p.execute()