import base64
import mimetypes
import os
from html import escape
from typing import Any, Callable, Dict, Iterable, Match, Optional, Tuple
import bs4
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from nbconvert.filters.strings import add_anchor
class MarkdownWithMath(Markdown):
    """Markdown text with math enabled."""
    DEFAULT_PLUGINS = ('strikethrough', 'table', 'url', 'task_lists', 'def_list')

    def __init__(self, renderer: HTMLRenderer, block: Optional[BlockParser]=None, inline: Optional[InlineParser]=None, plugins: Optional[Iterable[MarkdownPlugin]]=None):
        """Initialize the parser."""
        if block is None:
            block = MathBlockParser()
        if inline is None:
            if MISTUNE_V3:
                inline = MathInlineParser(hard_wrap=False)
            else:
                inline = MathInlineParser(renderer, hard_wrap=False)
        if plugins is None:
            plugins = (import_plugin(p) for p in self.DEFAULT_PLUGINS)
        super().__init__(renderer, block, inline, plugins)

    def render(self, source: str) -> str:
        """Render the HTML output for a Markdown source."""
        return str(super().__call__(source))