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
@attr.s(hash=False)
class BugBearChecker:
    name = 'flake8-bugbear'
    version = __version__
    tree = attr.ib(default=None)
    filename = attr.ib(default='(none)')
    lines = attr.ib(default=None)
    max_line_length = attr.ib(default=79)
    visitor = attr.ib(init=False, default=attr.Factory(lambda: BugBearVisitor))
    options = attr.ib(default=None)

    def run(self):
        if not self.tree or not self.lines:
            self.load_file()
        if self.options and hasattr(self.options, 'extend_immutable_calls'):
            b008_extend_immutable_calls = set(self.options.extend_immutable_calls)
        else:
            b008_extend_immutable_calls = set()
        b902_classmethod_decorators: set[str] = B902_default_decorators
        if self.options and hasattr(self.options, 'classmethod_decorators'):
            b902_classmethod_decorators = set(self.options.classmethod_decorators)
        visitor = self.visitor(filename=self.filename, lines=self.lines, b008_extend_immutable_calls=b008_extend_immutable_calls, b902_classmethod_decorators=b902_classmethod_decorators)
        visitor.visit(self.tree)
        for e in itertools.chain(visitor.errors, self.gen_line_based_checks()):
            if self.should_warn(e.message[:4]):
                yield self.adapt_error(e)

    def gen_line_based_checks(self):
        """gen_line_based_checks() -> (error, error, error, ...)

        The following simple checks are based on the raw lines, not the AST.
        """
        noqa_type_ignore_regex = re.compile('#\\s*(noqa|type:\\s*ignore)[^#\\r\\n]*$')
        for lineno, line in enumerate(self.lines, start=1):
            if lineno == 1 and line.startswith('#!'):
                continue
            no_comment_line = noqa_type_ignore_regex.sub('', line)
            if no_comment_line != line:
                no_comment_line = noqa_type_ignore_regex.sub('', no_comment_line)
            length = len(no_comment_line) - 1
            if length > 1.1 * self.max_line_length and no_comment_line.strip():
                chunks = no_comment_line.split()
                is_line_comment_url_path = len(chunks) == 2 and chunks[0] == '#'
                just_long_url_path = len(chunks) == 1
                num_leading_whitespaces = len(no_comment_line) - len(chunks[-1])
                too_many_leading_white_spaces = num_leading_whitespaces >= self.max_line_length - 7
                skip = is_line_comment_url_path or just_long_url_path
                if skip and (not too_many_leading_white_spaces):
                    continue
                yield B950(lineno, length, vars=(length, self.max_line_length))

    @classmethod
    def adapt_error(cls, e):
        """Adapts the extended error namedtuple to be compatible with Flake8."""
        return e._replace(message=e.message.format(*e.vars))[:4]

    def load_file(self):
        """Loads the file in a way that auto-detects source encoding and deals
        with broken terminal encodings for stdin.

        Stolen from flake8_import_order because it's good.
        """
        if self.filename in ('stdin', '-', None):
            self.filename = 'stdin'
            self.lines = pycodestyle.stdin_get_value().splitlines(True)
        else:
            self.lines = pycodestyle.readlines(self.filename)
        if not self.tree:
            self.tree = ast.parse(''.join(self.lines))

    @staticmethod
    def add_options(optmanager):
        """Informs flake8 to ignore B9xx by default."""
        optmanager.extend_default_ignore(disabled_by_default)
        optmanager.add_option('--extend-immutable-calls', comma_separated_list=True, parse_from_config=True, default=[], help='Skip B008 test for additional immutable calls.')
        if 'pep8ext_naming' not in sys.modules.keys():
            optmanager.add_option('--classmethod-decorators', comma_separated_list=True, parse_from_config=True, default=B902_default_decorators, help='List of method decorators that should be treated as classmethods by B902')

    @lru_cache
    def should_warn(self, code):
        """Returns `True` if Bugbear should emit a particular warning.

        flake8 overrides default ignores when the user specifies
        `ignore = ` in configuration.  This is problematic because it means
        specifying anything in `ignore = ` implicitly enables all optional
        warnings.  This function is a workaround for this behavior.

        As documented in the README, the user is expected to explicitly select
        the warnings.

        NOTE: This method is deprecated and will be removed in a future release. It is
        recommended to use `extend-ignore` and `extend-select` in your flake8
        configuration to avoid implicitly altering selected and ignored codes.
        """
        if code[:2] != 'B9':
            return True
        if self.options is None:
            LOG.info('Options not provided to Bugbear, optional warning %s selected.', code)
            return True
        for i in range(2, len(code) + 1):
            if self.options.select and code[:i] in self.options.select:
                return True
            if hasattr(self.options, 'extend_select') and self.options.extend_select and (code[:i] in self.options.extend_select):
                return True
        LOG.info('Optional warning %s not present in selected warnings: %r. Not firing it at all.', code, self.options.select)
        return False