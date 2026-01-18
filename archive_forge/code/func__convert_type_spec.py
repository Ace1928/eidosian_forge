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
def _convert_type_spec(_type: str, translations: Dict[str, str]={}) -> str:
    """Convert type specification to reference in reST."""
    if _type in translations:
        return translations[_type]
    elif _type == 'None':
        return ':obj:`None`'
    else:
        return ':class:`%s`' % _type
    return _type