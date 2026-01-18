import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
def _is_path_like(path):
    return isinstance(path, str) or hasattr(path, '__fspath__')