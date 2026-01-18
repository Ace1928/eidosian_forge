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
class PreRequest(Setting):
    name = 'pre_request'
    section = 'Server Hooks'
    validator = validate_callable(2)
    type = callable

    def pre_request(worker, req):
        worker.log.debug('%s %s', req.method, req.path)
    default = staticmethod(pre_request)
    desc = '        Called just before a worker processes the request.\n\n        The callable needs to accept two instance variables for the Worker and\n        the Request.\n        '