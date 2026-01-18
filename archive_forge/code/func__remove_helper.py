from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def _remove_helper(self, value, equal_func):

    def check_remove_node(node):
        for j in value:
            if equal_func(i, j):
                return True
        return False
    if not isinstance(value, list):
        value = [value]
    self._ensure_array_node()
    removed_list = []
    for i in self.node.args.arguments:
        if not check_remove_node(i):
            removed_list += [i]
    self.node.args.arguments = removed_list