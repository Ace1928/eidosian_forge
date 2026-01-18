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
def _fix_field_desc(self, desc: List[str]) -> List[str]:
    if self._is_list(desc):
        desc = [''] + desc
    elif desc[0].endswith('::'):
        desc_block = desc[1:]
        indent = self._get_indent(desc[0])
        block_indent = self._get_initial_indent(desc_block)
        if block_indent > indent:
            desc = [''] + desc
        else:
            desc = ['', desc[0]] + self._indent(desc_block, 4)
    return desc