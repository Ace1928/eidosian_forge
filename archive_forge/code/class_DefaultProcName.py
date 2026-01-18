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
class DefaultProcName(Setting):
    name = 'default_proc_name'
    section = 'Process Naming'
    validator = validate_string
    default = 'gunicorn'
    desc = '        Internal setting that is adjusted for each type of application.\n        '