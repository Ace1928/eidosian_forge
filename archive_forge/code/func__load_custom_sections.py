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
def _load_custom_sections(self) -> None:
    if self._config.napoleon_custom_sections is not None:
        for entry in self._config.napoleon_custom_sections:
            if isinstance(entry, str):
                self._sections[entry.lower()] = self._parse_custom_generic_section
            elif entry[1] == 'params_style':
                self._sections[entry[0].lower()] = self._parse_custom_params_style_section
            elif entry[1] == 'returns_style':
                self._sections[entry[0].lower()] = self._parse_custom_returns_style_section
            else:
                self._sections[entry[0].lower()] = self._sections.get(entry[1].lower(), self._parse_custom_generic_section)