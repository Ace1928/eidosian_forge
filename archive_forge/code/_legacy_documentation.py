import functools
import os
import pathlib
import types
import warnings
from typing import Union, Iterable, ContextManager, BinaryIO, TextIO, Any
from . import _common
A context manager providing a file path object to the resource.

    If the resource does not already exist on its own on the file system,
    a temporary file will be created. If the file was created, the file
    will be deleted upon exiting the context manager (no exception is
    raised if the file was deleted prior to the context manager
    exiting).
    