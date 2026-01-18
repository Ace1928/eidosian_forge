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
class Reload(Setting):
    name = 'reload'
    section = 'Debugging'
    cli = ['--reload']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Restart workers when code changes.\n\n        This setting is intended for development. It will cause workers to be\n        restarted whenever application code changes.\n\n        The reloader is incompatible with application preloading. When using a\n        paste configuration be sure that the server block does not import any\n        application code or the reload will not work as designed.\n\n        The default behavior is to attempt inotify with a fallback to file\n        system polling. Generally, inotify should be preferred if available\n        because it consumes less system resources.\n\n        .. note::\n           In order to use the inotify reloader, you must have the ``inotify``\n           package installed.\n        '