from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
class MachineFileParser:

    def __init__(self, filenames: T.List[str], sourcedir: str) -> None:
        self.parser = CmdLineFileParser()
        self.constants: T.Dict[str, T.Union[str, bool, int, T.List[str]]] = {'True': True, 'False': False}
        self.sections: T.Dict[str, T.Dict[str, T.Union[str, bool, int, T.List[str]]]] = {}
        for fname in filenames:
            with open(fname, encoding='utf-8') as f:
                content = f.read()
                content = content.replace('@GLOBAL_SOURCE_ROOT@', sourcedir)
                content = content.replace('@DIRNAME@', os.path.dirname(fname))
                try:
                    self.parser.read_string(content, fname)
                except configparser.Error as e:
                    raise EnvironmentException(f'Malformed machine file: {e}')
        if self.parser.has_section('constants'):
            self.constants.update(self._parse_section('constants'))
        for s in self.parser.sections():
            if s == 'constants':
                continue
            self.sections[s] = self._parse_section(s)

    def _parse_section(self, s: str) -> T.Dict[str, T.Union[str, bool, int, T.List[str]]]:
        self.scope = self.constants.copy()
        section: T.Dict[str, T.Union[str, bool, int, T.List[str]]] = {}
        for entry, value in self.parser.items(s):
            if ' ' in entry or '\t' in entry or "'" in entry or ('"' in entry):
                raise EnvironmentException(f'Malformed variable name {entry!r} in machine file.')
            value = value.replace('\\', '\\\\')
            try:
                ast = mparser.Parser(value, 'machinefile').parse()
                if not ast.lines:
                    raise EnvironmentException('value cannot be empty')
                res = self._evaluate_statement(ast.lines[0])
            except MesonException as e:
                raise EnvironmentException(f'Malformed value in machine file variable {entry!r}: {str(e)}.')
            except KeyError as e:
                raise EnvironmentException(f'Undefined constant {e.args[0]!r} in machine file variable {entry!r}.')
            section[entry] = res
            self.scope[entry] = res
        return section

    def _evaluate_statement(self, node: mparser.BaseNode) -> T.Union[str, bool, int, T.List[str]]:
        if isinstance(node, mparser.BaseStringNode):
            return node.value
        elif isinstance(node, mparser.BooleanNode):
            return node.value
        elif isinstance(node, mparser.NumberNode):
            return node.value
        elif isinstance(node, mparser.ParenthesizedNode):
            return self._evaluate_statement(node.inner)
        elif isinstance(node, mparser.ArrayNode):
            return [self._evaluate_statement(arg) for arg in node.args.arguments]
        elif isinstance(node, mparser.IdNode):
            return self.scope[node.value]
        elif isinstance(node, mparser.ArithmeticNode):
            l = self._evaluate_statement(node.left)
            r = self._evaluate_statement(node.right)
            if node.operation == 'add':
                if isinstance(l, str) and isinstance(r, str) or (isinstance(l, list) and isinstance(r, list)):
                    return l + r
            elif node.operation == 'div':
                if isinstance(l, str) and isinstance(r, str):
                    return os.path.join(l, r)
        raise EnvironmentException('Unsupported node type')