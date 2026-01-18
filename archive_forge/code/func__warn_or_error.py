import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
def _warn_or_error(msg):
    deprecation_allowed = ALLOW_DEPRECATION_IN_TEST in os.environ
    if _called_from_test() and (not deprecation_allowed):
        for filename, line_number, function_name, text in reversed(traceback.extract_stack()):
            if not _is_internal(filename) and (not filename.endswith(os.path.join('cirq', '_compat.py'))) and ('_test.py' in filename):
                break
        raise ValueError(f"During testing using Cirq deprecated functionality is not allowed: {msg}Update to non-deprecated functionality, or alternatively, you can quiet this error by removing the CIRQ_TESTING environment variable temporarily with `@mock.patch.dict(os.environ, clear='CIRQ_TESTING')`.\nIn case the usage of deprecated cirq is intentional, use `with cirq.testing.assert_deprecated(...):` around this line:\n{filename}:{line_number}: in {function_name}\n\t{text}")
    stack_level = 1
    for filename, _, _, _ in reversed(traceback.extract_stack()):
        if not _is_internal(filename) and '_compat.py' not in filename:
            break
        if '_compat.py' in filename:
            stack_level += 1
    warnings.warn(msg, DeprecationWarning, stacklevel=stack_level)