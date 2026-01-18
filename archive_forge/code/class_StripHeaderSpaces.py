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
class StripHeaderSpaces(Setting):
    name = 'strip_header_spaces'
    section = 'Server Mechanics'
    cli = ['--strip-header-spaces']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Strip spaces present between the header name and the the ``:``.\n\n        This is known to induce vulnerabilities and is not compliant with the HTTP/1.1 standard.\n        See https://portswigger.net/research/http-desync-attacks-request-smuggling-reborn.\n\n        Use with care and only if necessary.\n        '