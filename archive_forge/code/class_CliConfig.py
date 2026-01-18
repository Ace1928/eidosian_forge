import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size
import click
from redis import Redis
from redis.sentinel import Sentinel
from rq.defaults import (
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus
class CliConfig:
    """A helper class to be used with click commands, to handle shared options"""

    def __init__(self, url=None, config=None, worker_class=DEFAULT_WORKER_CLASS, job_class=DEFAULT_JOB_CLASS, death_penalty_class=DEFAULT_DEATH_PENALTY_CLASS, queue_class=DEFAULT_QUEUE_CLASS, connection_class=DEFAULT_CONNECTION_CLASS, path=None, *args, **kwargs):
        self._connection = None
        self.url = url
        self.config = config
        if path:
            for pth in path:
                sys.path.append(pth)
        try:
            self.worker_class = import_attribute(worker_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--worker-class')
        try:
            self.job_class = import_attribute(job_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--job-class')
        try:
            self.death_penalty_class = import_attribute(death_penalty_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--death-penalty-class')
        try:
            self.queue_class = import_attribute(queue_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--queue-class')
        try:
            self.connection_class = import_attribute(connection_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--connection-class')

    @property
    def connection(self):
        if self._connection is None:
            if self.url:
                self._connection = self.connection_class.from_url(self.url)
            elif self.config:
                settings = read_config_file(self.config) if self.config else {}
                self._connection = get_redis_from_config(settings, self.connection_class)
            else:
                self._connection = get_redis_from_config(os.environ, self.connection_class)
        return self._connection