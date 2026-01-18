import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
class IPythonDemo(Demo):
    """Class for interactive demos with IPython's input processing applied.

    This subclasses Demo, but instead of executing each block by the Python
    interpreter (via exec), it actually calls IPython on it, so that any input
    filters which may be in place are applied to the input block.

    If you have an interactive environment which exposes special input
    processing, you can use this class instead to write demo scripts which
    operate exactly as if you had typed them interactively.  The default Demo
    class requires the input to be valid, pure Python code.
    """

    def run_cell(self, source):
        """Execute a string with one or more lines of code"""
        self.shell.run_cell(source)