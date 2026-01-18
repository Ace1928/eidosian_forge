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
class WhenReady(Setting):
    name = 'when_ready'
    section = 'Server Hooks'
    validator = validate_callable(1)
    type = callable

    def when_ready(server):
        pass
    default = staticmethod(when_ready)
    desc = '        Called just after the server is started.\n\n        The callable needs to accept a single instance variable for the Arbiter.\n        '