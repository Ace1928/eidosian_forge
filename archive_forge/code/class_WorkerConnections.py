import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class WorkerConnections(Setting):
    name = 'worker_connections'
    section = 'Worker Processes'
    cli = ['--worker-connections']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 1000
    desc = '        The maximum number of simultaneous clients.\n\n        This setting only affects the ``gthread``, ``eventlet`` and ``gevent`` worker types.\n        '