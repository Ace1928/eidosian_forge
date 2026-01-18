from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
def _prepare_schema_patterns(schema: dict) -> dict[Pattern, str]:
    return {**_get_default_schema_selectors(), **{re.compile('^/' + selector + '$'): _default_schema_context for selector in schema.get(_lab_i18n_config, {}).get('selectors', [])}}