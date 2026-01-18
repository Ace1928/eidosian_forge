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
class HashErrors(InstallationError):
    """Multiple HashError instances rolled into one for reporting"""

    def __init__(self) -> None:
        self.errors: List['HashError'] = []

    def append(self, error: 'HashError') -> None:
        self.errors.append(error)

    def __str__(self) -> str:
        lines = []
        self.errors.sort(key=lambda e: e.order)
        for cls, errors_of_cls in groupby(self.errors, lambda e: e.__class__):
            lines.append(cls.head)
            lines.extend((e.body() for e in errors_of_cls))
        if lines:
            return '\n'.join(lines)
        return ''

    def __bool__(self) -> bool:
        return bool(self.errors)