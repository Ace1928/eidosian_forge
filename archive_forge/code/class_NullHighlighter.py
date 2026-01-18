import re
from abc import ABC, abstractmethod
from typing import List, Union
from .text import Span, Text
class NullHighlighter(Highlighter):
    """A highlighter object that doesn't highlight.

    May be used to disable highlighting entirely.

    """

    def highlight(self, text: Text) -> None:
        """Nothing to do"""