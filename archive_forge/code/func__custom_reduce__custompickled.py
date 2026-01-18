import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def _custom_reduce__custompickled(cp):
    return cp._reduce()