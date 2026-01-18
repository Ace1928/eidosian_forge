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
def _qualify_name(self, attr_name: str, klass: Type) -> str:
    warnings.warn('%s._qualify_name() is deprecated.' % self.__class__.__name__, RemovedInSphinx60Warning)
    if klass and '.' not in attr_name:
        if attr_name.startswith('~'):
            attr_name = attr_name[1:]
        try:
            q = klass.__qualname__
        except AttributeError:
            q = klass.__name__
        return '~%s.%s' % (q, attr_name)
    return attr_name