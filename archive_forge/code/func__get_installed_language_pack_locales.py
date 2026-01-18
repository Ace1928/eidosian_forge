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
def _get_installed_language_pack_locales() -> tuple[dict[str, Any], str]:
    """
    Get available installed language pack locales.

    Returns
    -------
    tuple
        A tuple, where the first item is the result and the second item any
        error messages.
    """
    data = {}
    messages = []
    for entry_point in entry_points(group=JUPYTERLAB_LANGUAGEPACK_ENTRY):
        try:
            data[entry_point.name] = os.path.dirname(entry_point.load().__file__)
        except Exception:
            messages.append(traceback.format_exc())
    message = '\n'.join(messages)
    return (data, message)