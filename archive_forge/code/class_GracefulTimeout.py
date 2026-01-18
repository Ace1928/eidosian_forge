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
class GracefulTimeout(Setting):
    name = 'graceful_timeout'
    section = 'Worker Processes'
    cli = ['--graceful-timeout']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 30
    desc = '        Timeout for graceful workers restart.\n\n        After receiving a restart signal, workers have this much time to finish\n        serving requests. Workers still alive after the timeout (starting from\n        the receipt of the restart signal) are force killed.\n        '