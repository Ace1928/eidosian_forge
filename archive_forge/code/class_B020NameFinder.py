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
class B020NameFinder(NameFinder):
    """Ignore names defined within the local scope of a comprehension."""

    def visit_GeneratorExp(self, node):
        self.visit(node.generators)

    def visit_ListComp(self, node):
        self.visit(node.generators)

    def visit_DictComp(self, node):
        self.visit(node.generators)

    def visit_comprehension(self, node):
        self.visit(node.iter)

    def visit_Lambda(self, node):
        self.visit(node.body)
        for lambda_arg in node.args.args:
            self.names.pop(lambda_arg.arg, None)