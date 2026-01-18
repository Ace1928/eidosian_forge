import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def PyCapsule_IsValid(capsule, name):
    return ctypes.pythonapi.PyCapsule_IsValid(ctypes.py_object(capsule), name) == 1