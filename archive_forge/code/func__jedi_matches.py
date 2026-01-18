from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
def _jedi_matches(self, cursor_column: int, cursor_line: int, text: str) -> Iterator[_JediCompletionLike]:
    """
        Return a list of :any:`jedi.api.Completion`\\s object from a ``text`` and
        cursor position.

        Parameters
        ----------
        cursor_column : int
            column position of the cursor in ``text``, 0-indexed.
        cursor_line : int
            line position of the cursor in ``text``, 0-indexed
        text : str
            text to complete

        Notes
        -----
        If ``IPCompleter.debug`` is ``True`` may return a :any:`_FakeJediCompletion`
        object containing a string with the Jedi debug information attached.

        .. deprecated:: 8.6
            You can use :meth:`_jedi_matcher` instead.
        """
    namespaces = [self.namespace]
    if self.global_namespace is not None:
        namespaces.append(self.global_namespace)
    completion_filter = lambda x: x
    offset = cursor_to_position(text, cursor_line, cursor_column)
    if offset:
        pre = text[offset - 1]
        if pre == '.':
            if self.omit__names == 2:
                completion_filter = lambda c: not c.name.startswith('_')
            elif self.omit__names == 1:
                completion_filter = lambda c: not (c.name.startswith('__') and c.name.endswith('__'))
            elif self.omit__names == 0:
                completion_filter = lambda x: x
            else:
                raise ValueError("Don't understand self.omit__names == {}".format(self.omit__names))
    interpreter = jedi.Interpreter(text[:offset], namespaces)
    try_jedi = True
    try:
        completing_string = False
        try:
            first_child = next((c for c in interpreter._get_module().tree_node.children if hasattr(c, 'value')))
        except StopIteration:
            pass
        else:
            completing_string = len(first_child.value) > 0 and first_child.value[0] in {"'", '"'}
        try_jedi = not completing_string
    except Exception as e:
        if self.debug:
            print('Error detecting if completing a non-finished string :', e, '|')
    if not try_jedi:
        return iter([])
    try:
        return filter(completion_filter, interpreter.complete(column=cursor_column, line=cursor_line + 1))
    except Exception as e:
        if self.debug:
            return iter([_FakeJediCompletion('Oops Jedi has crashed, please report a bug with the following:\n"""\n%s\ns"""' % e)])
        else:
            return iter([])