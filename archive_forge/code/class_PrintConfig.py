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
class PrintConfig(Setting):
    name = 'print_config'
    section = 'Debugging'
    cli = ['--print-config']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Print the configuration settings as fully resolved. Implies :ref:`check-config`.\n        '