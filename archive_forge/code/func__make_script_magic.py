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
def _make_script_magic(self, name):
    """make a named magic, that calls %%script with a particular program"""
    script = self.script_paths.get(name, name)

    @magic_arguments.magic_arguments()
    @script_args
    def named_script_magic(line, cell):
        if line:
            line = '%s %s' % (script, line)
        else:
            line = script
        return self.shebang(line, cell)
    named_script_magic.__doc__ = '%%{name} script magic\n        \n        Run cells with {script} in a subprocess.\n        \n        This is a shortcut for `%%script {script}`\n        '.format(**locals())
    return named_script_magic