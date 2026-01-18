import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def add_define_macros(self, macros):
    """Add define macros to configuration

        Add the given sequence of macro name and value duples to the beginning
        of the define_macros list This list will be visible to all extension
        modules of the current package.
        """
    dist = self.get_distribution()
    if dist is not None:
        if not hasattr(dist, 'define_macros'):
            dist.define_macros = []
        dist.define_macros.extend(macros)
    else:
        self.define_macros.extend(macros)