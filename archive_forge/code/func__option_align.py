import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def _option_align(arg):
    return directives.choice(arg, ('top', 'middle', 'bottom', 'left', 'center', 'right'))