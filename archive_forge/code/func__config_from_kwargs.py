from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def _config_from_kwargs(config: TextFixerConfig, kwargs: Dict[str, Any]) -> TextFixerConfig:
    """
    Handle parameters provided as keyword arguments to ftfy's top-level
    functions, converting them into a TextFixerConfig.
    """
    if 'fix_entities' in kwargs:
        warnings.warn('`fix_entities` has been renamed to `unescape_html`', DeprecationWarning)
        kwargs = kwargs.copy()
        kwargs['unescape_html'] = kwargs['fix_entities']
        del kwargs['fix_entities']
    config = config._replace(**kwargs)
    return config