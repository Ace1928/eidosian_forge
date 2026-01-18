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
def _parse_raises_section(self, section: str) -> List[str]:
    fields = self._consume_fields(parse_type=False, prefer_type=True)
    lines: List[str] = []
    for _name, _type, _desc in fields:
        m = self._name_rgx.match(_type)
        if m and m.group('name'):
            _type = m.group('name')
        elif _xref_regex.match(_type):
            pos = _type.find('`')
            _type = _type[pos + 1:-1]
        _type = ' ' + _type if _type else ''
        _desc = self._strip_empty(_desc)
        _descs = ' ' + '\n    '.join(_desc) if any(_desc) else ''
        lines.append(':raises%s:%s' % (_type, _descs))
    if lines:
        lines.append('')
    return lines