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
@staticmethod
def _translate_schema_strings(translations: Any, schema: dict, prefix: str='', to_translate: dict[Pattern, str] | None=None) -> None:
    """Translate a schema in-place."""
    if to_translate is None:
        to_translate = _prepare_schema_patterns(schema)
    for key, value in schema.items():
        path = prefix + '/' + key
        if isinstance(value, str):
            matched = False
            for pattern, context in to_translate.items():
                if pattern.fullmatch(path):
                    matched = True
                    break
            if matched:
                schema[key] = translations.pgettext(context, value)
        elif isinstance(value, dict):
            translator._translate_schema_strings(translations, value, prefix=path, to_translate=to_translate)
        elif isinstance(value, list):
            for i, element in enumerate(value):
                if not isinstance(element, dict):
                    continue
                translator._translate_schema_strings(translations, element, prefix=path + '[' + str(i) + ']', to_translate=to_translate)