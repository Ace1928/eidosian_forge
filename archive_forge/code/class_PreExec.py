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
class PreExec(Setting):
    name = 'pre_exec'
    section = 'Server Hooks'
    validator = validate_callable(1)
    type = callable

    def pre_exec(server):
        pass
    default = staticmethod(pre_exec)
    desc = '        Called just before a new master process is forked.\n\n        The callable needs to accept a single instance variable for the Arbiter.\n        '