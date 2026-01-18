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
class ConfigCheck(Setting):
    name = 'check_config'
    section = 'Debugging'
    cli = ['--check-config']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Check the configuration and exit. The exit status is 0 if the\n        configuration is correct, and 1 if the configuration is incorrect.\n        '