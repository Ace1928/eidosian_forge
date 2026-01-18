import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
class AutoPageArgumentParser(argparse.ArgumentParser, _ActionsContainer):

    @_substitute_formatter
    def _get_formatter(self) -> _HelpFormatter:
        return super()._get_formatter()