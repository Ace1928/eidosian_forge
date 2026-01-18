import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def _option_graph2use(arg):
    return directives.choice(arg, ('hierarchical', 'colored', 'flat', 'orig', 'exec'))