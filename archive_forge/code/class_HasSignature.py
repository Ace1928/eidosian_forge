from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
class HasSignature(object):
    """This is the class docstring."""
    __signature__ = Signature([Parameter('test', Parameter.POSITIONAL_OR_KEYWORD)])

    def __init__(self, *args):
        """This is the init docstring"""