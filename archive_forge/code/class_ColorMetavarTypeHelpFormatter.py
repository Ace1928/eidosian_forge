import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
class ColorMetavarTypeHelpFormatter(ColorHelpFormatter, argparse.MetavarTypeHelpFormatter):
    """Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')"""