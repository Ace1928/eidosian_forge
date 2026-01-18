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
def _parse_returns_section(self, section: str) -> List[str]:
    fields = self._consume_returns_section()
    multi = len(fields) > 1
    use_rtype = False if multi else self._config.napoleon_use_rtype
    lines: List[str] = []
    for _name, _type, _desc in fields:
        if use_rtype:
            field = self._format_field(_name, '', _desc)
        else:
            field = self._format_field(_name, _type, _desc)
        if multi:
            if lines:
                lines.extend(self._format_block('          * ', field))
            else:
                lines.extend(self._format_block(':returns: * ', field))
        else:
            if any(field):
                lines.extend(self._format_block(':returns: ', field))
            if _type and use_rtype:
                lines.extend([':rtype: %s' % _type, ''])
    if lines and lines[-1]:
        lines.append('')
    return lines