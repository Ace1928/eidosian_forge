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
class HashMissing(HashError):
    """A hash was needed for a requirement but is absent."""
    order = 2
    head = 'Hashes are required in --require-hashes mode, but they are missing from some requirements. Here is a list of those requirements along with the hashes their downloaded archives actually had. Add lines like these to your requirements files to prevent tampering. (If you did not enable --require-hashes manually, note that it turns on automatically when any package has a hash.)'

    def __init__(self, gotten_hash: str) -> None:
        """
        :param gotten_hash: The hash of the (possibly malicious) archive we
            just downloaded
        """
        self.gotten_hash = gotten_hash

    def body(self) -> str:
        from pip._internal.utils.hashes import FAVORITE_HASH
        package = None
        if self.req:
            package = self.req.original_link if self.req.is_direct else getattr(self.req, 'req', None)
        return '    {} --hash={}:{}'.format(package or 'unknown package', FAVORITE_HASH, self.gotten_hash)