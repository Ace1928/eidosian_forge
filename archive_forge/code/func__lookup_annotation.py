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
def _lookup_annotation(self, _name: str) -> str:
    if self._config.napoleon_attr_annotations:
        if self._what in ('module', 'class', 'exception') and self._obj:
            if not hasattr(self, '_annotations'):
                localns = getattr(self._config, 'autodoc_type_aliases', {})
                localns.update(getattr(self._config, 'napoleon_type_aliases', {}) or {})
                self._annotations = get_type_hints(self._obj, None, localns)
            if _name in self._annotations:
                return stringify_annotation(self._annotations[_name])
    return ''