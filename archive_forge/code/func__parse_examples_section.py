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
def _parse_examples_section(self, section: str) -> List[str]:
    labels = {'example': _('Example'), 'examples': _('Examples')}
    use_admonition = self._config.napoleon_use_admonition_for_examples
    label = labels.get(section.lower(), section)
    return self._parse_generic_section(label, use_admonition)