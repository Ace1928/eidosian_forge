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
class DogstatsdTags(Setting):
    name = 'dogstatsd_tags'
    section = 'Logging'
    cli = ['--dogstatsd-tags']
    meta = 'DOGSTATSD_TAGS'
    default = ''
    validator = validate_string
    desc = '    A comma-delimited list of datadog statsd (dogstatsd) tags to append to\n    statsd metrics.\n\n    .. versionadded:: 20\n    '