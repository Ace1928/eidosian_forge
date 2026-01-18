from __future__ import annotations
from contextlib import contextmanager
import locale
import platform
import re
import subprocess
from typing import TYPE_CHECKING
from pandas._config.config import options
def _valid_locales(locales: list[str] | str, normalize: bool) -> list[str]:
    """
    Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    return [loc for loc in (locale.normalize(loc.strip()) if normalize else loc.strip() for loc in locales) if can_set_locale(loc)]