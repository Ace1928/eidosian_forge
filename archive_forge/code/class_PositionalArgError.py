import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError
class PositionalArgError(AutocommandError):
    """
    Postional Arg Error: autocommand can't handle postional-only parameters
    """