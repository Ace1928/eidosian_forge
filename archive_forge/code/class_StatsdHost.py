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
class StatsdHost(Setting):
    name = 'statsd_host'
    section = 'Logging'
    cli = ['--statsd-host']
    meta = 'STATSD_ADDR'
    default = None
    validator = validate_statsd_address
    desc = '    The address of the StatsD server to log to.\n\n    Address is a string of the form:\n\n    * ``unix://PATH`` : for a unix domain socket.\n    * ``HOST:PORT`` : for a network address\n\n    .. versionadded:: 19.1\n    '