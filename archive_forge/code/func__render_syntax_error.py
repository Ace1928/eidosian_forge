from __future__ import absolute_import
import linecache
import os
import platform
import sys
from dataclasses import dataclass, field
from traceback import walk_tb
from types import ModuleType, TracebackType
from typing import (
from pip._vendor.pygments.lexers import guess_lexer_for_filename
from pip._vendor.pygments.token import Comment, Keyword, Name, Number, Operator, String
from pip._vendor.pygments.token import Text as TextToken
from pip._vendor.pygments.token import Token
from pip._vendor.pygments.util import ClassNotFound
from . import pretty
from ._loop import loop_last
from .columns import Columns
from .console import Console, ConsoleOptions, ConsoleRenderable, RenderResult, group
from .constrain import Constrain
from .highlighter import RegexHighlighter, ReprHighlighter
from .panel import Panel
from .scope import render_scope
from .style import Style
from .syntax import Syntax
from .text import Text
from .theme import Theme
@group()
def _render_syntax_error(self, syntax_error: _SyntaxError) -> RenderResult:
    highlighter = ReprHighlighter()
    path_highlighter = PathHighlighter()
    if syntax_error.filename != '<stdin>':
        if os.path.exists(syntax_error.filename):
            text = Text.assemble((f' {syntax_error.filename}', 'pygments.string'), (':', 'pygments.text'), (str(syntax_error.lineno), 'pygments.number'), style='pygments.text')
            yield path_highlighter(text)
    syntax_error_text = highlighter(syntax_error.line.rstrip())
    syntax_error_text.no_wrap = True
    offset = min(syntax_error.offset - 1, len(syntax_error_text))
    syntax_error_text.stylize('bold underline', offset, offset)
    syntax_error_text += Text.from_markup('\n' + ' ' * offset + '[traceback.offset]▲[/]', style='pygments.text')
    yield syntax_error_text