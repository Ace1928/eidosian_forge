import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class ReturnsSection(_KVSection):
    """Parser for numpydoc returns sections.

    E.g. any section that looks like this:
        return_name : type
            A description of this returned value
        another_type
            Return names are optional, types are required
    """
    is_generator = False

    def _parse_item(self, key: str, value: str) -> DocstringReturns:
        match = RETURN_KEY_REGEX.match(key)
        if match is not None:
            return_name = match.group('name')
            type_name = match.group('type')
        else:
            return_name = None
            type_name = None
        return DocstringReturns(args=[self.key], description=_clean_str(value), type_name=type_name, is_generator=self.is_generator, return_name=return_name)