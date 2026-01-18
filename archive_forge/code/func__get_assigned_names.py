from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
def _get_assigned_names(self, loop_node):
    loop_targets = (ast.For, ast.AsyncFor, ast.comprehension)
    for node in children_in_scope(loop_node):
        if isinstance(node, ast.Assign):
            for child in node.targets:
                yield from names_from_assignments(child)
        if isinstance(node, loop_targets + (ast.AnnAssign, ast.AugAssign)):
            yield from names_from_assignments(node.target)