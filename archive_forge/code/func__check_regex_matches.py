from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
@staticmethod
def _check_regex_matches(node, regex: str) -> bool:
    if isinstance(node, BaseStringNode):
        return re.match(regex, node.value) is not None
    return False