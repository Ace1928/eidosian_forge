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
def _deduped_module_warn_or_error(old_module_name: str, new_module_name: str, deadline: str):
    if _should_dedupe_module_deprecation() and old_module_name in _warned:
        return
    _warned.add(old_module_name)
    _warn_or_error(f'{old_module_name} was used but is deprecated.\n it will be removed in cirq {deadline}.\n Use {new_module_name} instead.\n')