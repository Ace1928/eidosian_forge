import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _exception_printer(exc):
    if _get_running_interactive_framework() in ['headless', None]:
        raise exc
    else:
        traceback.print_exc()