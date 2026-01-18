import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
def _get_context_value():
    return context_value.get()