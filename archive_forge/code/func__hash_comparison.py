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
def _hash_comparison(self) -> str:
    """
        Return a comparison of actual and expected hash values.

        Example::

               Expected sha256 abcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcde
                            or 123451234512345123451234512345123451234512345
                    Got        bcdefbcdefbcdefbcdefbcdefbcdefbcdefbcdefbcdef

        """

    def hash_then_or(hash_name: str) -> 'chain[str]':
        return chain([hash_name], repeat('    or'))
    lines: List[str] = []
    for hash_name, expecteds in self.allowed.items():
        prefix = hash_then_or(hash_name)
        lines.extend((f'        Expected {next(prefix)} {e}' for e in expecteds))
        lines.append(f'             Got        {self.gots[hash_name].hexdigest()}\n')
    return '\n'.join(lines)