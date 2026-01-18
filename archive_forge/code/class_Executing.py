import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
class Executing(object):
    """
    Information about the operation a frame is currently executing.

    Generally you will just want `node`, which is the AST node being executed,
    or None if it's unknown.

    If a decorator is currently being called, then:
        - `node` is a function or class definition
        - `decorator` is the expression in `node.decorator_list` being called
        - `statements == {node}`
    """

    def __init__(self, frame, source, node, stmts, decorator):
        self.frame = frame
        self.source = source
        self.node = node
        self.statements = stmts
        self.decorator = decorator

    def code_qualname(self):
        return self.source.code_qualname(self.frame.f_code)

    def text(self):
        return self.source._asttext_base().get_text(self.node)

    def text_range(self):
        return self.source._asttext_base().get_text_range(self.node)