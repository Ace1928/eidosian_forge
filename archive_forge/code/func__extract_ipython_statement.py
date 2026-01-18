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
def _extract_ipython_statement(stmt):
    while not isinstance(stmt.parent, ast.Module):
        stmt = stmt.parent
    tree = ast.parse('')
    tree.body = [cast(ast.stmt, stmt)]
    ast.copy_location(tree, stmt)
    return tree