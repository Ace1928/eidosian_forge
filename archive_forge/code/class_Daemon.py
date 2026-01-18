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
class Daemon(Setting):
    name = 'daemon'
    section = 'Server Mechanics'
    cli = ['-D', '--daemon']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Daemonize the Gunicorn process.\n\n        Detaches the server from the controlling terminal and enters the\n        background.\n        '