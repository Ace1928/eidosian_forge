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
class StatsdPrefix(Setting):
    name = 'statsd_prefix'
    section = 'Logging'
    cli = ['--statsd-prefix']
    meta = 'STATSD_PREFIX'
    default = ''
    validator = validate_string
    desc = '    Prefix to use when emitting statsd metrics (a trailing ``.`` is added,\n    if not provided).\n\n    .. versionadded:: 19.2\n    '