import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict

        Return True if this callback is using dictionary or nested groupings for
        Output dependencies.
        