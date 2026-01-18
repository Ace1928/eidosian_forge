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
class ProxyProtocol(Setting):
    name = 'proxy_protocol'
    section = 'Server Mechanics'
    cli = ['--proxy-protocol']
    validator = validate_bool
    default = False
    action = 'store_true'
    desc = '        Enable detect PROXY protocol (PROXY mode).\n\n        Allow using HTTP and Proxy together. It may be useful for work with\n        stunnel as HTTPS frontend and Gunicorn as HTTP server.\n\n        PROXY protocol: http://haproxy.1wt.eu/download/1.5/doc/proxy-protocol.txt\n\n        Example for stunnel config::\n\n            [https]\n            protocol = proxy\n            accept  = 443\n            connect = 80\n            cert = /etc/ssl/certs/stunnel.pem\n            key = /etc/ssl/certs/stunnel.key\n        '