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
def _consume_field(self, parse_type: bool=True, prefer_type: bool=False) -> Tuple[str, str, List[str]]:
    line = self._lines.next()
    if parse_type:
        _name, _, _type = self._partition_field_on_colon(line)
    else:
        _name, _type = (line, '')
    _name, _type = (_name.strip(), _type.strip())
    _name = self._escape_args_and_kwargs(_name)
    if parse_type and (not _type):
        _type = self._lookup_annotation(_name)
    if prefer_type and (not _type):
        _type, _name = (_name, _type)
    if self._config.napoleon_preprocess_types:
        _type = _convert_numpy_type_spec(_type, location=self._get_location(), translations=self._config.napoleon_type_aliases or {})
    indent = self._get_indent(line) + 1
    _desc = self._dedent(self._consume_indented_block(indent))
    _desc = self.__class__(_desc, self._config).lines()
    return (_name, _type, _desc)