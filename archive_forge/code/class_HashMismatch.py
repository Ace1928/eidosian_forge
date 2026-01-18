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
class HashMismatch(HashError):
    """
    Distribution file hash values don't match.

    :ivar package_name: The name of the package that triggered the hash
        mismatch. Feel free to write to this after the exception is raise to
        improve its error message.

    """
    order = 4
    head = 'THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. If you have updated the package versions, please update the hashes. Otherwise, examine the package contents carefully; someone may have tampered with them.'

    def __init__(self, allowed: Dict[str, List[str]], gots: Dict[str, '_Hash']) -> None:
        """
        :param allowed: A dict of algorithm names pointing to lists of allowed
            hex digests
        :param gots: A dict of algorithm names pointing to hashes we
            actually got from the files under suspicion
        """
        self.allowed = allowed
        self.gots = gots

    def body(self) -> str:
        return f'    {self._requirement_name()}:\n{self._hash_comparison()}'

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