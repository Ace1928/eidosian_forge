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
class ProxyAllowFrom(Setting):
    name = 'proxy_allow_ips'
    section = 'Server Mechanics'
    cli = ['--proxy-allow-from']
    validator = validate_string_to_list
    default = '127.0.0.1'
    desc = "        Front-end's IPs from which allowed accept proxy requests (comma separate).\n\n        Set to ``*`` to disable checking of Front-end IPs (useful for setups\n        where you don't know in advance the IP address of Front-end, but\n        you still trust the environment)\n        "