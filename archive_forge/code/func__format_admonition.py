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
def _format_admonition(self, admonition: str, lines: List[str]) -> List[str]:
    lines = self._strip_empty(lines)
    if len(lines) == 1:
        return ['.. %s:: %s' % (admonition, lines[0].strip()), '']
    elif lines:
        lines = self._indent(self._dedent(lines), 3)
        return ['.. %s::' % admonition, ''] + lines + ['']
    else:
        return ['.. %s::' % admonition, '']