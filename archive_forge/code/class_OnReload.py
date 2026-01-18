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
class OnReload(Setting):
    name = 'on_reload'
    section = 'Server Hooks'
    validator = validate_callable(1)
    type = callable

    def on_reload(server):
        pass
    default = staticmethod(on_reload)
    desc = '        Called to recycle workers during a reload via SIGHUP.\n\n        The callable needs to accept a single instance variable for the Arbiter.\n        '