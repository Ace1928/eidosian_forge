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
class SyslogTo(Setting):
    name = 'syslog_addr'
    section = 'Logging'
    cli = ['--log-syslog-to']
    meta = 'SYSLOG_ADDR'
    validator = validate_string
    if PLATFORM == 'darwin':
        default = 'unix:///var/run/syslog'
    elif PLATFORM in ('freebsd', 'dragonfly'):
        default = 'unix:///var/run/log'
    elif PLATFORM == 'openbsd':
        default = 'unix:///dev/log'
    else:
        default = 'udp://localhost:514'
    desc = '    Address to send syslog messages.\n\n    Address is a string of the form:\n\n    * ``unix://PATH#TYPE`` : for unix domain socket. ``TYPE`` can be ``stream``\n      for the stream driver or ``dgram`` for the dgram driver.\n      ``stream`` is the default.\n    * ``udp://HOST:PORT`` : for UDP sockets\n    * ``tcp://HOST:PORT`` : for TCP sockets\n\n    '