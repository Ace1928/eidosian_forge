import json
import numbers
from collections import OrderedDict
from functools import update_wrapper
from pprint import pformat
from typing import Any
import click
from click import Context, ParamType
from kombu.utils.objects import cached_property
from celery._state import get_current_app
from celery.signals import user_preload_options
from celery.utils import text
from celery.utils.log import mlevel
from celery.utils.time import maybe_iso8601
class CeleryDaemonCommand(CeleryCommand):
    """Daemon commands."""

    def __init__(self, *args, **kwargs):
        """Initialize a Celery command with common daemon options."""
        super().__init__(*args, **kwargs)
        self.params.extend((DaemonOption('--logfile', '-f', help='Log destination; defaults to stderr'), DaemonOption('--pidfile', help='PID file path; defaults to no PID file'), DaemonOption('--uid', help='Drops privileges to this user ID'), DaemonOption('--gid', help='Drops privileges to this group ID'), DaemonOption('--umask', help='Create files and directories with this umask'), DaemonOption('--executable', help='Override path to the Python executable')))