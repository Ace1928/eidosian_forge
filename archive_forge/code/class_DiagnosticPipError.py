import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class DiagnosticPipError(PipError):
    """An error, that presents diagnostic information to the user.

    This contains a bunch of logic, to enable pretty presentation of our error
    messages. Each error gets a unique reference. Each error can also include
    additional context, a hint and/or a note -- which are presented with the
    main error message in a consistent style.

    This is adapted from the error output styling in `sphinx-theme-builder`.
    """
    reference: str

    def __init__(self, *, kind: 'Literal["error", "warning"]'='error', reference: Optional[str]=None, message: Union[str, Text], context: Optional[Union[str, Text]], hint_stmt: Optional[Union[str, Text]], note_stmt: Optional[Union[str, Text]]=None, link: Optional[str]=None) -> None:
        if reference is None:
            assert hasattr(self, 'reference'), 'error reference not provided!'
            reference = self.reference
        assert _is_kebab_case(reference), 'error reference must be kebab-case!'
        self.kind = kind
        self.reference = reference
        self.message = message
        self.context = context
        self.note_stmt = note_stmt
        self.hint_stmt = hint_stmt
        self.link = link
        super().__init__(f'<{self.__class__.__name__}: {self.reference}>')

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}(reference={self.reference!r}, message={self.message!r}, context={self.context!r}, note_stmt={self.note_stmt!r}, hint_stmt={self.hint_stmt!r})>'

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        colour = 'red' if self.kind == 'error' else 'yellow'
        yield f'[{colour} bold]{self.kind}[/]: [bold]{self.reference}[/]'
        yield ''
        if not options.ascii_only:
            if self.context is not None:
                yield _prefix_with_indent(self.message, console, prefix=f'[{colour}]×[/] ', indent=f'[{colour}]│[/] ')
                yield _prefix_with_indent(self.context, console, prefix=f'[{colour}]╰─>[/] ', indent=f'[{colour}]   [/] ')
            else:
                yield _prefix_with_indent(self.message, console, prefix='[red]×[/] ', indent='  ')
        else:
            yield self.message
            if self.context is not None:
                yield ''
                yield self.context
        if self.note_stmt is not None or self.hint_stmt is not None:
            yield ''
        if self.note_stmt is not None:
            yield _prefix_with_indent(self.note_stmt, console, prefix='[magenta bold]note[/]: ', indent='      ')
        if self.hint_stmt is not None:
            yield _prefix_with_indent(self.hint_stmt, console, prefix='[cyan bold]hint[/]: ', indent='      ')
        if self.link is not None:
            yield ''
            yield f'Link: {self.link}'