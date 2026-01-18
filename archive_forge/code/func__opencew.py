import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re
import numpy as np
from ase.formula import formula_hill, formula_metal
def _opencew(filename, world=None):
    if world is None:
        from ase.parallel import world
    closelater = []

    def opener(file, flags):
        return os.open(file, flags | CEW_FLAGS)
    try:
        error = 0
        if world.rank == 0:
            try:
                fd = open(filename, 'wb', opener=opener)
            except OSError as ex:
                error = ex.errno
            else:
                closelater.append(fd)
        else:
            fd = open(os.devnull, 'wb')
            closelater.append(fd)
        error = world.sum(error)
        if error == errno.EEXIST:
            return None
        if error:
            raise OSError(error, 'Error', filename)
        return fd
    except BaseException:
        for fd in closelater:
            fd.close()
        raise