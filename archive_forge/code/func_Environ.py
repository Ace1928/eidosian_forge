import contextlib
import os
import sys
import random
import string
def Environ(**changes):
    """A context manager to temporarily change the os.environ"""
    return NoNoneDictMutator(os.environ, **changes)