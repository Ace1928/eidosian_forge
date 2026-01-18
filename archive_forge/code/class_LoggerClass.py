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
class LoggerClass(Setting):
    name = 'logger_class'
    section = 'Logging'
    cli = ['--logger-class']
    meta = 'STRING'
    validator = validate_class
    default = 'gunicorn.glogging.Logger'
    desc = '        The logger you want to use to log events in Gunicorn.\n\n        The default class (``gunicorn.glogging.Logger``) handles most\n        normal usages in logging. It provides error and access logging.\n\n        You can provide your own logger by giving Gunicorn a Python path to a\n        class that quacks like ``gunicorn.glogging.Logger``.\n        '