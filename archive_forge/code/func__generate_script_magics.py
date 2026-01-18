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
def _generate_script_magics(self):
    cell_magics = self.magics['cell']
    for name in self.script_magics:
        cell_magics[name] = self._make_script_magic(name)