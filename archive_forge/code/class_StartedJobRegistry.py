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
class StartedJobRegistry(BaseRegistry):
    """
    Registry of currently executing jobs. Each queue maintains a
    StartedJobRegistry. Jobs in this registry are ones that are currently
    being executed.

    Jobs are added to registry right before they are executed and removed
    right after completion (success or failure).
    """
    key_template = 'rq:wip:{0}'

    def cleanup(self, timestamp: Optional[float]=None):
        """Remove abandoned jobs from registry and add them to FailedJobRegistry.

        Removes jobs with an expiry time earlier than timestamp, specified as
        seconds since the Unix epoch. timestamp defaults to call time if
        unspecified. Removed jobs are added to the global failed job queue.

        Args:
            timestamp (datetime): The datetime to use as the limit.
        """
        score = timestamp if timestamp is not None else current_timestamp()
        job_ids = self.get_expired_job_ids(score)
        if job_ids:
            failed_job_registry = FailedJobRegistry(self.name, self.connection, serializer=self.serializer)
            with self.connection.pipeline() as pipeline:
                for job_id in job_ids:
                    try:
                        job = self.job_class.fetch(job_id, connection=self.connection, serializer=self.serializer)
                    except NoSuchJobError:
                        continue
                    job.execute_failure_callback(self.death_penalty_class, AbandonedJobError, AbandonedJobError(), traceback.extract_stack())
                    retry = job.retries_left and job.retries_left > 0
                    if retry:
                        queue = self.get_queue()
                        job.retry(queue, pipeline)
                    else:
                        exc_string = f'due to {AbandonedJobError.__name__}'
                        logger.warning(f'{self.__class__.__name__} cleanup: Moving job to {FailedJobRegistry.__name__} ({exc_string})')
                        job.set_status(JobStatus.FAILED)
                        job._exc_info = f'Moved to {FailedJobRegistry.__name__}, {exc_string}, at {datetime.now()}'
                        job.save(pipeline=pipeline, include_meta=False)
                        job.cleanup(ttl=-1, pipeline=pipeline)
                        failed_job_registry.add(job, job.failure_ttl)
                pipeline.zremrangebyscore(self.key, 0, score)
                pipeline.execute()
        return job_ids