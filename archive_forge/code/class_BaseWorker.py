import contextlib
import errno
import logging
import math
import os
import random
import signal
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from enum import Enum
from random import shuffle
from types import FrameType
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from contextlib import suppress
import redis.exceptions
from . import worker_registration
from .command import PUBSUB_CHANNEL_TEMPLATE, handle_command, parse_payload
from .connections import get_current_connection, pop_connection, push_connection
from .defaults import (
from .exceptions import DequeueTimeout, DeserializationError, ShutDownImminentException
from .job import Job, JobStatus
from .logutils import blue, green, setup_loghandlers, yellow
from .maintenance import clean_intermediate_queue
from .queue import Queue
from .registry import StartedJobRegistry, clean_registries
from .scheduler import RQScheduler
from .serializers import resolve_serializer
from .suspension import is_suspended
from .timeouts import HorseMonitorTimeoutException, JobTimeoutException, UnixSignalDeathPenalty
from .utils import as_text, backend_class, compact, ensure_list, get_version, utcformat, utcnow, utcparse
from .version import VERSION
class BaseWorker:
    redis_worker_namespace_prefix = 'rq:worker:'
    redis_workers_keys = worker_registration.REDIS_WORKER_KEYS
    death_penalty_class = UnixSignalDeathPenalty
    queue_class = Queue
    job_class = Job
    log_result_lifespan = True
    log_job_description = True
    exponential_backoff_factor = 2.0
    max_connection_wait_time = 60.0

    def __init__(self, queues, name: Optional[str]=None, default_result_ttl=DEFAULT_RESULT_TTL, connection: Optional['Redis']=None, exc_handler=None, exception_handlers=None, default_worker_ttl=DEFAULT_WORKER_TTL, maintenance_interval: int=DEFAULT_MAINTENANCE_TASK_INTERVAL, job_class: Optional[Type['Job']]=None, queue_class: Optional[Type['Queue']]=None, log_job_description: bool=True, job_monitoring_interval=DEFAULT_JOB_MONITORING_INTERVAL, disable_default_exception_handler: bool=False, prepare_for_work: bool=True, serializer=None, work_horse_killed_handler: Optional[Callable[[Job, int, int, 'struct_rusage'], None]]=None):
        self.default_result_ttl = default_result_ttl
        self.worker_ttl = default_worker_ttl
        self.job_monitoring_interval = job_monitoring_interval
        self.maintenance_interval = maintenance_interval
        connection = self._set_connection(connection)
        self.connection = connection
        self.redis_server_version = None
        self.job_class = backend_class(self, 'job_class', override=job_class)
        self.queue_class = backend_class(self, 'queue_class', override=queue_class)
        self.version = VERSION
        self.python_version = sys.version
        self.serializer = resolve_serializer(serializer)
        queues = [self.queue_class(name=q, connection=connection, job_class=self.job_class, serializer=self.serializer, death_penalty_class=self.death_penalty_class) if isinstance(q, str) else q for q in ensure_list(queues)]
        self.name: str = name or uuid4().hex
        self.queues = queues
        self.validate_queues()
        self._ordered_queues = self.queues[:]
        self._exc_handlers: List[Callable] = []
        self._work_horse_killed_handler = work_horse_killed_handler
        self._shutdown_requested_date: Optional[datetime] = None
        self._state: str = 'starting'
        self._is_horse: bool = False
        self._horse_pid: int = 0
        self._stop_requested: bool = False
        self._stopped_job_id = None
        self.log = logger
        self.log_job_description = log_job_description
        self.last_cleaned_at = None
        self.successful_job_count: int = 0
        self.failed_job_count: int = 0
        self.total_working_time: int = 0
        self.current_job_working_time: float = 0
        self.birth_date = None
        self.scheduler: Optional[RQScheduler] = None
        self.pubsub = None
        self.pubsub_thread = None
        self._dequeue_strategy: DequeueStrategy = DequeueStrategy.DEFAULT
        self.disable_default_exception_handler = disable_default_exception_handler
        if prepare_for_work:
            self.hostname: Optional[str] = socket.gethostname()
            self.pid: Optional[int] = os.getpid()
            try:
                connection.client_setname(self.name)
            except redis.exceptions.ResponseError:
                warnings.warn('CLIENT SETNAME command not supported, setting ip_address to unknown', Warning)
                self.ip_address = 'unknown'
            else:
                client_adresses = [client['addr'] for client in connection.client_list() if client['name'] == self.name]
                if len(client_adresses) > 0:
                    self.ip_address = client_adresses[0]
                else:
                    warnings.warn('CLIENT LIST command not supported, setting ip_address to unknown', Warning)
                    self.ip_address = 'unknown'
        else:
            self.hostname = None
            self.pid = None
            self.ip_address = 'unknown'
        if isinstance(exception_handlers, (list, tuple)):
            for handler in exception_handlers:
                self.push_exc_handler(handler)
        elif exception_handlers is not None:
            self.push_exc_handler(exception_handlers)

    @classmethod
    def all(cls, connection: Optional['Redis']=None, job_class: Optional[Type['Job']]=None, queue_class: Optional[Type['Queue']]=None, queue: Optional['Queue']=None, serializer=None) -> List['Worker']:
        """Returns an iterable of all Workers.

        Returns:
            workers (List[Worker]): A list of workers
        """
        if queue:
            connection = queue.connection
        elif connection is None:
            connection = get_current_connection()
        worker_keys = worker_registration.get_keys(queue=queue, connection=connection)
        workers = [cls.find_by_key(key, connection=connection, job_class=job_class, queue_class=queue_class, serializer=serializer) for key in worker_keys]
        return compact(workers)

    @classmethod
    def all_keys(cls, connection: Optional['Redis']=None, queue: Optional['Queue']=None) -> List[str]:
        """List of worker keys

        Args:
            connection (Optional[Redis], optional): A Redis Connection. Defaults to None.
            queue (Optional[Queue], optional): The Queue. Defaults to None.

        Returns:
            list_keys (List[str]): A list of worker keys
        """
        return [as_text(key) for key in worker_registration.get_keys(queue=queue, connection=connection)]

    @classmethod
    def count(cls, connection: Optional['Redis']=None, queue: Optional['Queue']=None) -> int:
        """Returns the number of workers by queue or connection.

        Args:
            connection (Optional[Redis], optional): Redis connection. Defaults to None.
            queue (Optional[Queue], optional): The queue to use. Defaults to None.

        Returns:
            length (int): The queue length.
        """
        return len(worker_registration.get_keys(queue=queue, connection=connection))

    @property
    def should_run_maintenance_tasks(self):
        """Maintenance tasks should run on first startup or every 10 minutes."""
        if self.last_cleaned_at is None:
            return True
        if utcnow() - self.last_cleaned_at > timedelta(seconds=self.maintenance_interval):
            return True
        return False

    def _set_connection(self, connection: Optional['Redis']) -> 'Redis':
        """Configures the Redis connection to have a socket timeout.
        This should timouet the connection in case any specific command hangs at any given time (eg. BLPOP).
        If the connection provided already has a `socket_timeout` defined, skips.

        Args:
            connection (Optional[Redis]): The Redis Connection.
        """
        if connection is None:
            connection = get_current_connection()
        current_socket_timeout = connection.connection_pool.connection_kwargs.get('socket_timeout')
        if current_socket_timeout is None:
            timeout_config = {'socket_timeout': self.connection_timeout}
            connection.connection_pool.connection_kwargs.update(timeout_config)
        return connection

    @property
    def dequeue_timeout(self) -> int:
        return max(1, self.worker_ttl - 15)

    def clean_registries(self):
        """Runs maintenance jobs on each Queue's registries."""
        for queue in self.queues:
            if queue.acquire_maintenance_lock():
                self.log.info('Cleaning registries for queue: %s', queue.name)
                clean_registries(queue)
                worker_registration.clean_worker_registry(queue)
                clean_intermediate_queue(self, queue)
                queue.release_maintenance_lock()
        self.last_cleaned_at = utcnow()

    def get_redis_server_version(self):
        """Return Redis server version of connection"""
        if not self.redis_server_version:
            self.redis_server_version = get_version(self.connection)
        return self.redis_server_version

    def validate_queues(self):
        """Sanity check for the given queues."""
        for queue in self.queues:
            if not isinstance(queue, self.queue_class):
                raise TypeError('{0} is not of type {1} or string types'.format(queue, self.queue_class))

    def queue_names(self) -> List[str]:
        """Returns the queue names of this worker's queues.

        Returns:
            List[str]: The queue names.
        """
        return [queue.name for queue in self.queues]

    def queue_keys(self) -> List[str]:
        """Returns the Redis keys representing this worker's queues.

        Returns:
            List[str]: The list of strings with queues keys
        """
        return [queue.key for queue in self.queues]

    @property
    def key(self):
        """Returns the worker's Redis hash key."""
        return self.redis_worker_namespace_prefix + self.name

    @property
    def pubsub_channel_name(self):
        """Returns the worker's Redis hash key."""
        return PUBSUB_CHANNEL_TEMPLATE % self.name

    @property
    def supports_redis_streams(self) -> bool:
        """Only supported by Redis server >= 5.0 is required."""
        return self.get_redis_server_version() >= (5, 0, 0)

    def _install_signal_handlers(self):
        """Installs signal handlers for handling SIGINT and SIGTERM gracefully."""
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def work(self, burst: bool=False, logging_level: str='INFO', date_format: str=DEFAULT_LOGGING_DATE_FORMAT, log_format: str=DEFAULT_LOGGING_FORMAT, max_jobs: Optional[int]=None, max_idle_time: Optional[int]=None, with_scheduler: bool=False, dequeue_strategy: DequeueStrategy=DequeueStrategy.DEFAULT) -> bool:
        """Starts the work loop.

        Pops and performs all jobs on the current list of queues.  When all
        queues are empty, block and wait for new jobs to arrive on any of the
        queues, unless `burst` mode is enabled.
        If `max_idle_time` is provided, worker will die when it's idle for more than the provided value.

        The return value indicates whether any jobs were processed.

        Args:
            burst (bool, optional): Whether to work on burst mode. Defaults to False.
            logging_level (str, optional): Logging level to use. Defaults to "INFO".
            date_format (str, optional): Date Format. Defaults to DEFAULT_LOGGING_DATE_FORMAT.
            log_format (str, optional): Log Format. Defaults to DEFAULT_LOGGING_FORMAT.
            max_jobs (Optional[int], optional): Max number of jobs. Defaults to None.
            max_idle_time (Optional[int], optional): Max seconds for worker to be idle. Defaults to None.
            with_scheduler (bool, optional): Whether to run the scheduler in a separate process. Defaults to False.
            dequeue_strategy (DequeueStrategy, optional): Which strategy to use to dequeue jobs.
                Defaults to DequeueStrategy.DEFAULT

        Returns:
            worked (bool): Will return True if any job was processed, False otherwise.
        """
        self.bootstrap(logging_level, date_format, log_format)
        self._dequeue_strategy = dequeue_strategy
        completed_jobs = 0
        if with_scheduler:
            self._start_scheduler(burst, logging_level, date_format, log_format)
        self._install_signal_handlers()
        try:
            while True:
                try:
                    self.check_for_suspension(burst)
                    if self.should_run_maintenance_tasks:
                        self.run_maintenance_tasks()
                    if self._stop_requested:
                        self.log.info('Worker %s: stopping on request', self.key)
                        break
                    timeout = None if burst else self.dequeue_timeout
                    result = self.dequeue_job_and_maintain_ttl(timeout, max_idle_time)
                    if result is None:
                        if burst:
                            self.log.info('Worker %s: done, quitting', self.key)
                        elif max_idle_time is not None:
                            self.log.info('Worker %s: idle for %d seconds, quitting', self.key, max_idle_time)
                        break
                    job, queue = result
                    self.execute_job(job, queue)
                    self.heartbeat()
                    completed_jobs += 1
                    if max_jobs is not None:
                        if completed_jobs >= max_jobs:
                            self.log.info('Worker %s: finished executing %d jobs, quitting', self.key, completed_jobs)
                            break
                except redis.exceptions.TimeoutError:
                    self.log.error('Worker %s: Redis connection timeout, quitting...', self.key)
                    break
                except StopRequested:
                    break
                except SystemExit:
                    raise
                except:
                    self.log.error('Worker %s: found an unhandled exception, quitting...', self.key, exc_info=True)
                    break
        finally:
            self.teardown()
        return bool(completed_jobs)

    def handle_job_failure(self, job: 'Job', queue: 'Queue', started_job_registry=None, exc_string=''):
        """
        Handles the failure or an executing job by:
            1. Setting the job status to failed
            2. Removing the job from StartedJobRegistry
            3. Setting the workers current job to None
            4. Add the job to FailedJobRegistry
        `save_exc_to_job` should only be used for testing purposes
        """
        self.log.debug('Handling failed execution of job %s', job.id)
        with self.connection.pipeline() as pipeline:
            if started_job_registry is None:
                started_job_registry = StartedJobRegistry(job.origin, self.connection, job_class=self.job_class, serializer=self.serializer)
            job_is_stopped = self._stopped_job_id == job.id
            retry = job.retries_left and job.retries_left > 0 and (not job_is_stopped)
            if job_is_stopped:
                job.set_status(JobStatus.STOPPED, pipeline=pipeline)
                self._stopped_job_id = None
            elif not retry:
                job.set_status(JobStatus.FAILED, pipeline=pipeline)
            started_job_registry.remove(job, pipeline=pipeline)
            if not self.disable_default_exception_handler and (not retry):
                job._handle_failure(exc_string, pipeline=pipeline)
                with suppress(redis.exceptions.ConnectionError):
                    pipeline.execute()
            self.set_current_job_id(None, pipeline=pipeline)
            self.increment_failed_job_count(pipeline)
            if job.started_at and job.ended_at:
                self.increment_total_working_time(job.ended_at - job.started_at, pipeline)
            if retry:
                job.retry(queue, pipeline)
                enqueue_dependents = False
            else:
                enqueue_dependents = True
            try:
                pipeline.execute()
                if enqueue_dependents:
                    queue.enqueue_dependents(job)
            except Exception:
                pass

    def _start_scheduler(self, burst: bool=False, logging_level: str='INFO', date_format: str=DEFAULT_LOGGING_DATE_FORMAT, log_format: str=DEFAULT_LOGGING_FORMAT):
        """Starts the scheduler process.
        This is specifically designed to be run by the worker when running the `work()` method.
        Instanciates the RQScheduler and tries to acquire a lock.
        If the lock is acquired, start scheduler.
        If worker is on burst mode just enqueues scheduled jobs and quits,
        otherwise, starts the scheduler in a separate process.

        Args:
            burst (bool, optional): Whether to work on burst mode. Defaults to False.
            logging_level (str, optional): Logging level to use. Defaults to "INFO".
            date_format (str, optional): Date Format. Defaults to DEFAULT_LOGGING_DATE_FORMAT.
            log_format (str, optional): Log Format. Defaults to DEFAULT_LOGGING_FORMAT.
        """
        self.scheduler = RQScheduler(self.queues, connection=self.connection, logging_level=logging_level, date_format=date_format, log_format=log_format, serializer=self.serializer)
        self.scheduler.acquire_locks()
        if self.scheduler.acquired_locks:
            if burst:
                self.scheduler.enqueue_scheduled_jobs()
                self.scheduler.release_locks()
            else:
                self.scheduler.start()

    def bootstrap(self, logging_level: str='INFO', date_format: str=DEFAULT_LOGGING_DATE_FORMAT, log_format: str=DEFAULT_LOGGING_FORMAT):
        """Bootstraps the worker.
        Runs the basic tasks that should run when the worker actually starts working.
        Used so that new workers can focus on the work loop implementation rather
        than the full bootstraping process.

        Args:
            logging_level (str, optional): Logging level to use. Defaults to "INFO".
            date_format (str, optional): Date Format. Defaults to DEFAULT_LOGGING_DATE_FORMAT.
            log_format (str, optional): Log Format. Defaults to DEFAULT_LOGGING_FORMAT.
        """
        setup_loghandlers(logging_level, date_format, log_format)
        self.register_birth()
        self.log.info('Worker %s started with PID %d, version %s', self.key, os.getpid(), VERSION)
        self.subscribe()
        self.set_state(WorkerStatus.STARTED)
        qnames = self.queue_names()
        self.log.info('*** Listening on %s...', green(', '.join(qnames)))

    def check_for_suspension(self, burst: bool):
        """Check to see if workers have been suspended by `rq suspend`"""
        before_state = None
        notified = False
        while not self._stop_requested and is_suspended(self.connection, self):
            if burst:
                self.log.info('Suspended in burst mode, exiting')
                self.log.info('Note: There could still be unfinished jobs on the queue')
                raise StopRequested
            if not notified:
                self.log.info('Worker suspended, run `rq resume` to resume')
                before_state = self.get_state()
                self.set_state(WorkerStatus.SUSPENDED)
                notified = True
            time.sleep(1)
        if before_state:
            self.set_state(before_state)

    def run_maintenance_tasks(self):
        """
        Runs periodic maintenance tasks, these include:
        1. Check if scheduler should be started. This check should not be run
           on first run since worker.work() already calls
           `scheduler.enqueue_scheduled_jobs()` on startup.
        2. Cleaning registries

        No need to try to start scheduler on first run
        """
        if self.last_cleaned_at:
            if self.scheduler and (not self.scheduler._process or not self.scheduler._process.is_alive()):
                self.scheduler.acquire_locks(auto_start=True)
        self.clean_registries()

    def subscribe(self):
        """Subscribe to this worker's channel"""
        self.log.info('Subscribing to channel %s', self.pubsub_channel_name)
        self.pubsub = self.connection.pubsub()
        self.pubsub.subscribe(**{self.pubsub_channel_name: self.handle_payload})
        self.pubsub_thread = self.pubsub.run_in_thread(sleep_time=0.2, daemon=True)

    def unsubscribe(self):
        """Unsubscribe from pubsub channel"""
        if self.pubsub_thread:
            self.log.info('Unsubscribing from channel %s', self.pubsub_channel_name)
            self.pubsub_thread.stop()
            self.pubsub_thread.join()
            self.pubsub.unsubscribe()
            self.pubsub.close()

    def dequeue_job_and_maintain_ttl(self, timeout: Optional[int], max_idle_time: Optional[int]=None) -> Tuple['Job', 'Queue']:
        """Dequeues a job while maintaining the TTL.

        Returns:
            result (Tuple[Job, Queue]): A tuple with the job and the queue.
        """
        result = None
        qnames = ','.join(self.queue_names())
        self.set_state(WorkerStatus.IDLE)
        self.procline('Listening on ' + qnames)
        self.log.debug('*** Listening on %s...', green(qnames))
        connection_wait_time = 1.0
        idle_since = utcnow()
        idle_time_left = max_idle_time
        while True:
            try:
                self.heartbeat()
                if self.should_run_maintenance_tasks:
                    self.run_maintenance_tasks()
                if timeout is not None and idle_time_left is not None:
                    timeout = min(timeout, idle_time_left)
                self.log.debug('Dequeueing jobs on queues %s and timeout %s', green(qnames), timeout)
                result = self.queue_class.dequeue_any(self._ordered_queues, timeout, connection=self.connection, job_class=self.job_class, serializer=self.serializer, death_penalty_class=self.death_penalty_class)
                if result is not None:
                    job, queue = result
                    self.reorder_queues(reference_queue=queue)
                    self.log.debug('Dequeued job %s from %s', blue(job.id), green(queue.name))
                    job.redis_server_version = self.get_redis_server_version()
                    if self.log_job_description:
                        self.log.info('%s: %s (%s)', green(queue.name), blue(job.description), job.id)
                    else:
                        self.log.info('%s: %s', green(queue.name), job.id)
                break
            except DequeueTimeout:
                if max_idle_time is not None:
                    idle_for = (utcnow() - idle_since).total_seconds()
                    idle_time_left = math.ceil(max_idle_time - idle_for)
                    if idle_time_left <= 0:
                        break
            except redis.exceptions.ConnectionError as conn_err:
                self.log.error('Could not connect to Redis instance: %s Retrying in %d seconds...', conn_err, connection_wait_time)
                time.sleep(connection_wait_time)
                connection_wait_time *= self.exponential_backoff_factor
                connection_wait_time = min(connection_wait_time, self.max_connection_wait_time)
            else:
                connection_wait_time = 1.0
        self.heartbeat()
        return result

    def heartbeat(self, timeout: Optional[int]=None, pipeline: Optional['Pipeline']=None):
        """Specifies a new worker timeout, typically by extending the
        expiration time of the worker, effectively making this a "heartbeat"
        to not expire the worker until the timeout passes.

        The next heartbeat should come before this time, or the worker will
        die (at least from the monitoring dashboards).

        If no timeout is given, the worker_ttl will be used to update
        the expiration time of the worker.

        Args:
            timeout (Optional[int]): Timeout
            pipeline (Optional[Redis]): A Redis pipeline
        """
        timeout = timeout or self.worker_ttl + 60
        connection: Union[Redis, 'Pipeline'] = pipeline if pipeline is not None else self.connection
        connection.expire(self.key, timeout)
        connection.hset(self.key, 'last_heartbeat', utcformat(utcnow()))
        self.log.debug('Sent heartbeat to prevent worker timeout. Next one should arrive in %s seconds.', timeout)