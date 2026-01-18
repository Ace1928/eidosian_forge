import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
class IPythonLineDemo(IPythonDemo, LineDemo):
    """Variant of the LineDemo class whose input is processed by IPython."""
    pass