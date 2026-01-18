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
class CACerts(Setting):
    name = 'ca_certs'
    section = 'SSL'
    cli = ['--ca-certs']
    meta = 'FILE'
    validator = validate_string
    default = None
    desc = '    CA certificates file\n    '