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
class Procname(Setting):
    name = 'proc_name'
    section = 'Process Naming'
    cli = ['-n', '--name']
    meta = 'STRING'
    validator = validate_string
    default = None
    desc = "        A base to use with setproctitle for process naming.\n\n        This affects things like ``ps`` and ``top``. If you're going to be\n        running more than one instance of Gunicorn you'll probably want to set a\n        name to tell them apart. This requires that you install the setproctitle\n        module.\n\n        If not set, the *default_proc_name* setting will be used.\n        "