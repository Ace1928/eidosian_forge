import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
Make a name for a block of code, and cache the code.

        Parameters
        ----------
        transformed_code : str
            The executable Python source code to cache and compile.
        number : int
            A number which forms part of the code's name. Used for the execution
            counter.
        raw_code : str
            The raw code before transformation, if None, set to `transformed_code`.

        Returns
        -------
        The name of the cached code (as a string). Pass this as the filename
        argument to compilation, so that tracebacks are correctly hooked up.
        