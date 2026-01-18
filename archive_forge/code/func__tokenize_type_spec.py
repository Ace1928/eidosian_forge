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
def _tokenize_type_spec(spec: str) -> List[str]:

    def postprocess(item):
        if _default_regex.match(item):
            default = item[:7]
            other = item[8:]
            return [default, ' ', other]
        else:
            return [item]
    tokens = [item for raw_token in _token_regex.split(spec) for item in postprocess(raw_token) if item]
    return tokens