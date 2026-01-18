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
class DoHandshakeOnConnect(Setting):
    name = 'do_handshake_on_connect'
    section = 'SSL'
    cli = ['--do-handshake-on-connect']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = "    Whether to perform SSL handshake on socket connect (see stdlib ssl module's)\n    "