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
def _make_proxy_spec_property(source_module: ModuleType) -> property:

    def fget(self):
        return source_module.__spec__

    def fset(self, value):
        source_module.__spec__ = value
    return property(fget, fset)