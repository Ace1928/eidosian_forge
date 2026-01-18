from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def _raw_parse(self) -> None:
    """Parse the source to find the interesting facts about its lines.

        A handful of attributes are updated.

        """
    if self.exclude:
        self.raw_excluded = self.lines_matching(self.exclude)
    prev_toktype: int = token.INDENT
    indent: int = 0
    exclude_indent: int = 0
    excluding: bool = False
    excluding_decorators: bool = False
    first_line: int = 0
    empty: bool = True
    first_on_line: bool = True
    nesting: int = 0
    assert self.text is not None
    tokgen = generate_tokens(self.text)
    for toktype, ttext, (slineno, _), (elineno, _), ltext in tokgen:
        if self.show_tokens:
            print('%10s %5s %-20r %r' % (tokenize.tok_name.get(toktype, toktype), nice_pair((slineno, elineno)), ttext, ltext))
        if toktype == token.INDENT:
            indent += 1
        elif toktype == token.DEDENT:
            indent -= 1
        elif toktype == token.NAME:
            if ttext == 'class':
                self.raw_classdefs.add(slineno)
        elif toktype == token.OP:
            if ttext == ':' and nesting == 0:
                should_exclude = self.raw_excluded.intersection(range(first_line, elineno + 1)) or excluding_decorators
                if not excluding and should_exclude:
                    self.raw_excluded.add(elineno)
                    exclude_indent = indent
                    excluding = True
                    excluding_decorators = False
            elif ttext == '@' and first_on_line:
                if elineno in self.raw_excluded:
                    excluding_decorators = True
                if excluding_decorators:
                    self.raw_excluded.add(elineno)
            elif ttext in '([{':
                nesting += 1
            elif ttext in ')]}':
                nesting -= 1
        elif toktype == token.STRING:
            if prev_toktype == token.INDENT:
                self.raw_docstrings.update(range(slineno, elineno + 1))
        elif toktype == token.NEWLINE:
            if first_line and elineno != first_line:
                for l in range(first_line, elineno + 1):
                    self._multiline[l] = first_line
            first_line = 0
            first_on_line = True
        if ttext.strip() and toktype != tokenize.COMMENT:
            empty = False
            if not first_line:
                first_line = slineno
                if excluding and indent <= exclude_indent:
                    excluding = False
                if excluding:
                    self.raw_excluded.add(elineno)
                first_on_line = False
        prev_toktype = toktype
    if not empty:
        byte_parser = ByteParser(self.text, filename=self.filename)
        self.raw_statements.update(byte_parser._find_statements())
    if env.PYBEHAVIOR.module_firstline_1 and self._multiline:
        self._multiline[1] = min(self.raw_statements)