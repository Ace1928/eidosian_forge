import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def _get_min_indent(self, lines: List[str]) -> int:
    min_indent = None
    for line in lines:
        if line:
            indent = self._get_indent(line)
            if min_indent is None:
                min_indent = indent
            elif indent < min_indent:
                min_indent = indent
    return min_indent or 0