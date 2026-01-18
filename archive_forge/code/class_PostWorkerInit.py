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
class PostWorkerInit(Setting):
    name = 'post_worker_init'
    section = 'Server Hooks'
    validator = validate_callable(1)
    type = callable

    def post_worker_init(worker):
        pass
    default = staticmethod(post_worker_init)
    desc = '        Called just after a worker has initialized the application.\n\n        The callable needs to accept one instance variable for the initialized\n        Worker.\n        '