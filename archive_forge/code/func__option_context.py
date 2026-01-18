import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def _option_context(arg):
    if arg in [None, 'reset', 'close-figs']:
        return arg
    raise ValueError("argument should be None or 'reset' or 'close-figs'")