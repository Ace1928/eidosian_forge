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
class ChildExit(Setting):
    name = 'child_exit'
    section = 'Server Hooks'
    validator = validate_callable(2)
    type = callable

    def child_exit(server, worker):
        pass
    default = staticmethod(child_exit)
    desc = '        Called just after a worker has been exited, in the master process.\n\n        The callable needs to accept two instance variables for the Arbiter and\n        the just-exited Worker.\n\n        .. versionadded:: 19.7\n        '