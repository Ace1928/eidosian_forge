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
class ReusePort(Setting):
    name = 'reuse_port'
    section = 'Server Mechanics'
    cli = ['--reuse-port']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Set the ``SO_REUSEPORT`` flag on the listening socket.\n\n        .. versionadded:: 19.8\n        '