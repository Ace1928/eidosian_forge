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
@lru_cache
def _get_default_schema_selectors() -> dict[Pattern, str]:
    return {re.compile('^/' + pattern + '$'): context for pattern, context in DEFAULT_SCHEMA_SELECTORS.items()}