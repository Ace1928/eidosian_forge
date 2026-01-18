import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
class ColorRawDescriptionHelpFormatter(ColorHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Help message formatter which retains any formatting in descriptions."""