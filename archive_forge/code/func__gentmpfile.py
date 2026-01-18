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
@contextmanager
def _gentmpfile(suffix):
    try:
        tmpdir = mkdtemp()
        ntf = open(os.path.join(tmpdir, 'temp%s' % suffix), 'wt')
        yield ntf
    finally:
        try:
            ntf.close()
            os.remove(ntf)
        except:
            pass
        else:
            os.rmdir(tmpdir)