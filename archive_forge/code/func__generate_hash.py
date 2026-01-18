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
@staticmethod
def _generate_hash() -> str:
    """
        Generates a unique hash for the current state of the source code of the object.
        This hash is used to determine if the source code has changed since the last compilation.
        """
    source = getsource(self.obj)
    return hashlib.sha256(source.encode('utf-8')).hexdigest()