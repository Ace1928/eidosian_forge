import doctest
import re
import sys
import time
from io import StringIO
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
import sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relpath
from sphinx.util.typing import OptionSpec
class TestCode:

    def __init__(self, code: str, type: str, filename: str, lineno: int, options: Optional[Dict]=None) -> None:
        self.code = code
        self.type = type
        self.filename = filename
        self.lineno = lineno
        self.options = options or {}

    def __repr__(self) -> str:
        return 'TestCode(%r, %r, filename=%r, lineno=%r, options=%r)' % (self.code, self.type, self.filename, self.lineno, self.options)