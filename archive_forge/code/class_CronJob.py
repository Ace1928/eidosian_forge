from __future__ import annotations
import enum
import typing
import datetime
import croniter
from aiokeydb.v2.types.base import BaseModel, lazyproperty, Field, validator
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.configs import settings
from aiokeydb.v2.utils.logs import logger
from aiokeydb.v2.types.static import JobStatus, TaskType, TERMINAL_STATUSES, UNSUCCESSFUL_TERMINAL_STATUSES, INCOMPLETE_STATUSES
class CronJob(BaseModel):
    """
    Allows scheduling of repeated jobs with cron syntax.

    function: the async function to run
    cron: cron string for a job to be repeated, uses croniter
    unique: unique jobs only one once per queue, defaults true

    Remaining kwargs are pass through to Job
    """
    function: typing.Callable
    cron: str
    unique: bool = True
    cron_name: typing.Optional[str] = None
    timeout: typing.Optional[int] = Field(default_factory=settings.get_default_job_timeout)
    retries: typing.Optional[int] = Field(default_factory=settings.get_default_job_retries)
    ttl: typing.Optional[int] = Field(default_factory=settings.get_default_job_ttl)
    heartbeat: typing.Optional[int] = None
    default_kwargs: typing.Optional[dict] = None
    callback: typing.Optional[typing.Union[str, typing.Callable]] = None
    callback_kwargs: typing.Optional[dict] = Field(default_factory=dict)
    bypass_lock: typing.Optional[bool] = None

    @property
    def function_name(self) -> str:
        """
        Returns the name of the function
        """
        return self.cron_name or self.function.__qualname__

    @validator('callback')
    def validate_callback(cls, v: typing.Optional[typing.Union[str, typing.Callable]]) -> typing.Optional[str]:
        """
        Validates the callback and returns the function name
        """
        return v if v is None else get_func_full_name(v)

    def next_scheduled(self) -> int:
        """
        Returns the next scheduled time for the cron job
        """
        return int(croniter.croniter(self.cron, seconds(now())).get_next())

    def to_enqueue_kwargs(self, job_key: typing.Optional[str]=None, exclude_none: typing.Optional[bool]=True, **kwargs) -> typing.Dict[str, typing.Any]:
        """
        Returns the kwargs for the job
        """
        default_kwargs = self.default_kwargs or {}
        if kwargs:
            default_kwargs.update(kwargs)
        default_kwargs['key'] = job_key
        enqueue_kwargs = {'job_or_func': self.function_name, **default_kwargs}
        if self.callback:
            enqueue_kwargs['job_callback'] = self.callback
            enqueue_kwargs['job_callback_kwargs'] = self.callback_kwargs
        if self.bypass_lock is not None:
            enqueue_kwargs['bypass_lock'] = self.bypass_lock
        if exclude_none:
            enqueue_kwargs = {k: v for k, v in enqueue_kwargs.items() if v is not None}
        enqueue_kwargs['scheduled'] = self.next_scheduled()
        return enqueue_kwargs