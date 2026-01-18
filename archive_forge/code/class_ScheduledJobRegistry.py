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
class ScheduledJobRegistry(BaseRegistry):
    """
    Registry of scheduled jobs.
    """
    key_template = 'rq:scheduled:{0}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_jobs_to_enqueue = self.get_expired_job_ids

    def schedule(self, job: 'Job', scheduled_datetime, pipeline: Optional['Pipeline']=None):
        """
        Adds job to registry, scored by its execution time (in UTC).
        If datetime has no tzinfo, it will assume localtimezone.
        """
        if not scheduled_datetime.tzinfo:
            tz = timezone(timedelta(seconds=-(time.timezone if time.daylight == 0 else time.altzone)))
            scheduled_datetime = scheduled_datetime.replace(tzinfo=tz)
        timestamp = calendar.timegm(scheduled_datetime.utctimetuple())
        return self.connection.zadd(self.key, {job.id: timestamp})

    def cleanup(self):
        """This method is only here to prevent errors because this method is
        automatically called by `count()` and `get_job_ids()` methods
        implemented in BaseRegistry."""
        pass

    def remove_jobs(self, timestamp: Optional[datetime]=None, pipeline: Optional['Pipeline']=None):
        """Remove jobs whose timestamp is in the past from registry.

        Args:
            timestamp (Optional[datetime], optional): The timestamp. Defaults to None.
            pipeline (Optional[Pipeline], optional): The Redis pipeline. Defaults to None.
        """
        connection = pipeline if pipeline is not None else self.connection
        score = timestamp if timestamp is not None else current_timestamp()
        return connection.zremrangebyscore(self.key, 0, score)

    def get_jobs_to_schedule(self, timestamp: Optional[datetime]=None, chunk_size: int=1000) -> List[str]:
        """Get's a list of job IDs that should be scheduled.

        Args:
            timestamp (Optional[datetime], optional): _description_. Defaults to None.
            chunk_size (int, optional): _description_. Defaults to 1000.

        Returns:
            jobs (List[str]): A list of Job ids
        """
        score = timestamp if timestamp is not None else current_timestamp()
        jobs_to_schedule = self.connection.zrangebyscore(self.key, 0, score, start=0, num=chunk_size)
        return [as_text(job_id) for job_id in jobs_to_schedule]

    def get_scheduled_time(self, job_or_id: Union['Job', str]) -> datetime:
        """Returns datetime (UTC) at which job is scheduled to be enqueued

        Args:
            job_or_id (Union[Job, str]): The Job instance or Job ID

        Raises:
            NoSuchJobError: If the job was not found

        Returns:
            datetime (datetime): The scheduled time as datetime object
        """
        if isinstance(job_or_id, self.job_class):
            job_id = job_or_id.id
        else:
            job_id = job_or_id
        score = self.connection.zscore(self.key, job_id)
        if not score:
            raise NoSuchJobError
        return datetime.fromtimestamp(score, tz=timezone.utc)