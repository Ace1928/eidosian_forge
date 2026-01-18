import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class _SphinxSection(Section):
    """Base parser for numpydoc sections with sphinx-style syntax.

    E.g. sections that look like this:
        .. title:: something
            possibly over multiple lines
    """

    @property
    def title_pattern(self) -> str:
        return f'^\\.\\.\\s*({self.title})\\s*::'