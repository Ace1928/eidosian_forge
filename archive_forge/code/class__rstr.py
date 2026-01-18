import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
class _rstr(str):
    """String that doesn't quote its repr."""

    def __repr__(self):
        return self