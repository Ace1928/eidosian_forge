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
class Paste(Setting):
    name = 'paste'
    section = 'Server Mechanics'
    cli = ['--paste', '--paster']
    meta = 'STRING'
    validator = validate_string
    default = None
    desc = '        Load a PasteDeploy config file. The argument may contain a ``#``\n        symbol followed by the name of an app section from the config file,\n        e.g. ``production.ini#admin``.\n\n        At this time, using alternate server blocks is not supported. Use the\n        command line arguments to control server configuration instead.\n        '