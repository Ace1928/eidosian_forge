import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
@staticmethod
def _make_method(func, instance):
    return types.MethodType(func, instance)