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
class Backlog(Setting):
    name = 'backlog'
    section = 'Server Socket'
    cli = ['--backlog']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 2048
    desc = '        The maximum number of pending connections.\n\n        This refers to the number of clients that can be waiting to be served.\n        Exceeding this number results in the client getting an error when\n        attempting to connect. It should only affect servers under significant\n        load.\n\n        Must be a positive integer. Generally set in the 64-2048 range.\n        '