from __future__ import annotations
import json
import re
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Iterable
import huggingface_hub
import semantic_version as semver
from gradio_client.documentation import document
from huggingface_hub import CommitOperationAdd
from gradio.themes.utils import (
from gradio.themes.utils.readme_content import README_CONTENT
def _get_computed_value(self, property: str, depth=0) -> str:
    max_depth = 100
    if depth > max_depth:
        warnings.warn(f"Cannot resolve '{property}' - circular reference detected.")
        return ''
    is_dark = property.endswith('_dark')
    if is_dark:
        set_value = getattr(self, property, getattr(self, property[:-5], ''))
    else:
        set_value = getattr(self, property, '')
    pattern = '(\\*)([\\w_]+)(\\b)'

    def repl_func(match, depth):
        word = match.group(2)
        dark_suffix = '_dark' if property.endswith('_dark') else ''
        return self._get_computed_value(word + dark_suffix, depth + 1)
    computed_value = re.sub(pattern, lambda match: repl_func(match, depth), set_value)
    return computed_value