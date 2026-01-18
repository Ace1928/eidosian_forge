import hashlib
import os
import asyncio
import aiofiles
from inspect import getsource, getmembers, isfunction, isclass, iscoroutinefunction
from Cython.Build import cythonize
from anyio import Path
from setuptools import setup, Extension
import sys
import logging
import logging.config
import pathlib
from typing import (
from indelogging import (
import concurrent_log_handler
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import (
from inspect import iscoroutinefunction
from functools import wraps
def _compile_cython(self) -> None:
    """
        Compiles the Python source code to a Cython C extension.
        """
    setup(ext_modules=cythonize(Extension(name=self.module_name, sources=[getsource(self.obj)])))