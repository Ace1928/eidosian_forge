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
def _partition_field_on_colon(self, line: str) -> Tuple[str, str, str]:
    before_colon = []
    after_colon = []
    colon = ''
    found_colon = False
    for i, source in enumerate(_xref_or_code_regex.split(line)):
        if found_colon:
            after_colon.append(source)
        else:
            m = _single_colon_regex.search(source)
            if i % 2 == 0 and m:
                found_colon = True
                colon = source[m.start():m.end()]
                before_colon.append(source[:m.start()])
                after_colon.append(source[m.end():])
            else:
                before_colon.append(source)
    return (''.join(before_colon).strip(), colon, ''.join(after_colon).strip())