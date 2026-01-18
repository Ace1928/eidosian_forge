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
def _parse_keyword_arguments_section(self, section: str) -> List[str]:
    fields = self._consume_fields()
    if self._config.napoleon_use_keyword:
        return self._format_docutils_params(fields, field_role='keyword', type_role='kwtype')
    else:
        return self._format_fields(_('Keyword Arguments'), fields)