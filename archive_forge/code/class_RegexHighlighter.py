import re
from abc import ABC, abstractmethod
from typing import List, Union
from .text import Span, Text
class RegexHighlighter(Highlighter):
    """Applies highlighting from a list of regular expressions."""
    highlights: List[str] = []
    base_style: str = ''

    def highlight(self, text: Text) -> None:
        """Highlight :class:`rich.text.Text` using regular expressions.

        Args:
            text (~Text): Text to highlighted.

        """
        highlight_regex = text.highlight_regex
        for re_highlight in self.highlights:
            highlight_regex(re_highlight, style_prefix=self.base_style)