import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
class ColorArgDefaultsHelpFormatter(ColorHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Help message formatter which adds default values to argument help."""