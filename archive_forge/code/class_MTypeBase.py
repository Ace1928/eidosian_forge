from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
class MTypeBase:

    def __init__(self, node: T.Optional[BaseNode]=None):
        if node is None:
            self.node = self.new_node()
        else:
            self.node = node
        self.node_type = None
        for i in self.supported_nodes():
            if isinstance(self.node, i):
                self.node_type = i

    @classmethod
    def new_node(cls, value=None):
        raise RewriterException('Internal error: new_node of MTypeBase was called')

    @classmethod
    def supported_nodes(cls):
        return []

    def can_modify(self):
        return self.node_type is not None

    def get_node(self):
        return self.node

    def add_value(self, value):
        mlog.warning('Cannot add a value of type', mlog.bold(type(self).__name__), '--> skipping')

    def remove_value(self, value):
        mlog.warning('Cannot remove a value of type', mlog.bold(type(self).__name__), '--> skipping')

    def remove_regex(self, value):
        mlog.warning('Cannot remove a regex in type', mlog.bold(type(self).__name__), '--> skipping')