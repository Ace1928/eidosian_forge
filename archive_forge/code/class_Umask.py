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
class Umask(Setting):
    name = 'umask'
    section = 'Server Mechanics'
    cli = ['-m', '--umask']
    meta = 'INT'
    validator = validate_pos_int
    type = auto_int
    default = 0
    desc = '        A bit mask for the file mode on files written by Gunicorn.\n\n        Note that this affects unix socket permissions.\n\n        A valid value for the ``os.umask(mode)`` call or a string compatible\n        with ``int(value, 0)`` (``0`` means Python guesses the base, so values\n        like ``0``, ``0xFF``, ``0022`` are valid for decimal, hex, and octal\n        representations)\n        '