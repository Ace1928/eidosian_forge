from __future__ import annotations
import sys
from typing import ClassVar, Dict, Iterable, List, Optional, Type, Union
from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.table import Table
from . import box
from ._loop import loop_first
from ._stack import Stack
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .containers import Renderables
from .jupyter import JupyterMixin
from .panel import Panel
from .rule import Rule
from .segment import Segment
from .style import Style, StyleStack
from .syntax import Syntax
from .text import Text, TextType
class MarkdownElement:
    new_line: ClassVar[bool] = True

    @classmethod
    def create(cls, markdown: 'Markdown', token: Token) -> 'MarkdownElement':
        """Factory to create markdown element,

        Args:
            markdown (Markdown): The parent Markdown object.
            token (Token): A node from markdown-it.

        Returns:
            MarkdownElement: A new markdown element
        """
        return cls()

    def on_enter(self, context: 'MarkdownContext') -> None:
        """Called when the node is entered.

        Args:
            context (MarkdownContext): The markdown context.
        """

    def on_text(self, context: 'MarkdownContext', text: TextType) -> None:
        """Called when text is parsed.

        Args:
            context (MarkdownContext): The markdown context.
        """

    def on_leave(self, context: 'MarkdownContext') -> None:
        """Called when the parser leaves the element.

        Args:
            context (MarkdownContext): [description]
        """

    def on_child_close(self, context: 'MarkdownContext', child: 'MarkdownElement') -> bool:
        """Called when a child element is closed.

        This method allows a parent element to take over rendering of its children.

        Args:
            context (MarkdownContext): The markdown context.
            child (MarkdownElement): The child markdown element.

        Returns:
            bool: Return True to render the element, or False to not render the element.
        """
        return True

    def __rich_console__(self, console: 'Console', options: 'ConsoleOptions') -> 'RenderResult':
        return ()