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
class LimitRequestFields(Setting):
    name = 'limit_request_fields'
    section = 'Security'
    cli = ['--limit-request-fields']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 100
    desc = "        Limit the number of HTTP headers fields in a request.\n\n        This parameter is used to limit the number of headers in a request to\n        prevent DDOS attack. Used with the *limit_request_field_size* it allows\n        more safety. By default this value is 100 and can't be larger than\n        32768.\n        "