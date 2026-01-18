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
def _parse_generic_section(self, section: str, use_admonition: bool) -> List[str]:
    lines = self._strip_empty(self._consume_to_next_section())
    lines = self._dedent(lines)
    if use_admonition:
        header = '.. admonition:: %s' % section
        lines = self._indent(lines, 3)
    else:
        header = '.. rubric:: %s' % section
    if lines:
        return [header, ''] + lines + ['']
    else:
        return [header, '']