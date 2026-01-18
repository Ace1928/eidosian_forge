import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted.