import asyncio
import asyncio.exceptions
import atexit
import errno
import os
import signal
import sys
import time
from subprocess import CalledProcessError
from threading import Thread
from traitlets import Any, Dict, List, default
from IPython.core import magic_arguments
from IPython.core.async_helpers import _AsyncIOProxy
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.utils.process import arg_split
@default('script_magics')
def _script_magics_default(self):
    """default to a common list of programs"""
    defaults = ['sh', 'bash', 'perl', 'ruby', 'python', 'python2', 'python3', 'pypy']
    if os.name == 'nt':
        defaults.extend(['cmd'])
    return defaults