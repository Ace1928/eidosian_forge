from __future__ import annotations
import gettext as gettext_module
import os.path
from threading import local
def _get_default_locale_path() -> str | None:
    try:
        if __file__ is None:
            return None
        return os.path.join(os.path.dirname(__file__), 'locale')
    except NameError:
        return None