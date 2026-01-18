import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
class MatchesIterator:
    """Stores a list of matches and which one is currently selected if any.

    Also responsible for doing the actual replacement of the original line with
    the selected match.

    A MatchesIterator can be `clear`ed to reset match iteration, and
    `update`ed to set what matches will be iterated over."""

    def __init__(self) -> None:
        self.current_word = ''
        self.matches: List[str] = []
        self.index = -1
        self.orig_cursor_offset = -1
        self.orig_line = ''
        self.completer: Optional[autocomplete.BaseCompletionType] = None
        self.start: Optional[int] = None
        self.end: Optional[int] = None

    def __nonzero__(self) -> bool:
        """MatchesIterator is False when word hasn't been replaced yet"""
        return self.index != -1

    def __bool__(self) -> bool:
        return self.index != -1

    @property
    def candidate_selected(self) -> bool:
        """True when word selected/replaced, False when word hasn't been
        replaced yet"""
        return bool(self)

    def __iter__(self) -> 'MatchesIterator':
        return self

    def current(self) -> str:
        if self.index == -1:
            raise ValueError('No current match.')
        return self.matches[self.index]

    def __next__(self) -> str:
        self.index = (self.index + 1) % len(self.matches)
        return self.matches[self.index]

    def previous(self) -> str:
        if self.index <= 0:
            self.index = len(self.matches)
        self.index -= 1
        return self.matches[self.index]

    def cur_line(self) -> Tuple[int, str]:
        """Returns a cursor offset and line with the current substitution
        made"""
        return self.substitute(self.current())

    def substitute(self, match: str) -> Tuple[int, str]:
        """Returns a cursor offset and line with match substituted in"""
        assert self.completer is not None
        lp = self.completer.locate(self.orig_cursor_offset, self.orig_line)
        assert lp is not None
        return (lp.start + len(match), self.orig_line[:lp.start] + match + self.orig_line[lp.stop:])

    def is_cseq(self) -> bool:
        return bool(os.path.commonprefix(self.matches)[len(self.current_word):])

    def substitute_cseq(self) -> Tuple[int, str]:
        """Returns a new line by substituting a common sequence in, and update
        matches"""
        assert self.completer is not None
        cseq = os.path.commonprefix(self.matches)
        new_cursor_offset, new_line = self.substitute(cseq)
        if len(self.matches) == 1:
            self.clear()
        else:
            self.update(new_cursor_offset, new_line, self.matches, self.completer)
            if len(self.matches) == 1:
                self.clear()
        return (new_cursor_offset, new_line)

    def update(self, cursor_offset: int, current_line: str, matches: List[str], completer: autocomplete.BaseCompletionType) -> None:
        """Called to reset the match index and update the word being replaced

        Should only be called if there's a target to update - otherwise, call
        clear"""
        if matches is None:
            raise ValueError('Matches may not be None.')
        self.orig_cursor_offset = cursor_offset
        self.orig_line = current_line
        self.matches = matches
        self.completer = completer
        self.index = -1
        lp = self.completer.locate(self.orig_cursor_offset, self.orig_line)
        assert lp is not None
        self.start = lp.start
        self.end = lp.stop
        self.current_word = lp.word

    def clear(self) -> None:
        self.matches = []
        self.orig_cursor_offset = -1
        self.orig_line = ''
        self.current_word = ''
        self.start = None
        self.end = None
        self.index = -1