import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
class GraphError(RuntimeError):
    pass