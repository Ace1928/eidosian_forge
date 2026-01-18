import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
@cache
def _frame_filter(name, filename):
    omit_functions = ['unwind::unwind', 'CapturedTraceback::gather', 'gather_with_cpp', '_start', '__libc_start_main', 'PyEval_', 'PyObject_', 'PyFunction_']
    omit_filenames = ['core/boxing', '/Register', '/Redispatch', 'pythonrun.c', 'Modules/main.c', 'Objects/call.c', 'Objects/methodobject.c', 'pycore_ceval.h', 'ceval.c', 'cpython/abstract.h']
    for of in omit_functions:
        if of in name:
            return False
    for of in omit_filenames:
        if of in filename:
            return False
    return True