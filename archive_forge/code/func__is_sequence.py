import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import numpy as np
import functools
import os
import subprocess
import sys
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path
def _is_sequence(arg):
    if isinstance(arg, (str, bytes)):
        return False
    try:
        len(arg)
        return True
    except Exception:
        return False