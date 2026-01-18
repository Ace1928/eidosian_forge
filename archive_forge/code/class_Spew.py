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
class Spew(Setting):
    name = 'spew'
    section = 'Debugging'
    cli = ['--spew']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Install a trace function that spews every line executed by the server.\n\n        This is the nuclear option.\n        '