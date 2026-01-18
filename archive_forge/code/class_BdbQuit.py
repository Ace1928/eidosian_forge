import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
class BdbQuit(Exception):
    """Exception to give up completely."""