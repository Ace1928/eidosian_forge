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
class Loglevel(Setting):
    name = 'loglevel'
    section = 'Logging'
    cli = ['--log-level']
    meta = 'LEVEL'
    validator = validate_string
    default = 'info'
    desc = "        The granularity of Error log outputs.\n\n        Valid level names are:\n\n        * ``'debug'``\n        * ``'info'``\n        * ``'warning'``\n        * ``'error'``\n        * ``'critical'``\n        "