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
class LogConfigJson(Setting):
    name = 'logconfig_json'
    section = 'Logging'
    cli = ['--log-config-json']
    meta = 'FILE'
    validator = validate_string
    default = None
    desc = '    The log config to read config from a JSON file\n\n    Format: https://docs.python.org/3/library/logging.config.html#logging.config.jsonConfig\n\n    .. versionadded:: 20.0\n    '