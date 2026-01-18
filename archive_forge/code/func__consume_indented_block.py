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
def _consume_indented_block(self, indent: int=1) -> List[str]:
    lines = []
    line = self._lines.get(0)
    while not self._is_section_break() and (not line or self._is_indented(line, indent)):
        lines.append(self._lines.next())
        line = self._lines.get(0)
    return lines