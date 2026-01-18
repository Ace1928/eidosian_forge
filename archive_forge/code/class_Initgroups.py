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
class Initgroups(Setting):
    name = 'initgroups'
    section = 'Server Mechanics'
    cli = ['--initgroups']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = "        If true, set the worker process's group access list with all of the\n        groups of which the specified username is a member, plus the specified\n        group id.\n\n        .. versionadded:: 19.7\n        "