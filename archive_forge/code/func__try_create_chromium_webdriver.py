from __future__ import annotations
import logging # isort:skip
from ..util.dependencies import import_required # isort:skip
import_required("selenium.webdriver",
import atexit
import os
from os.path import devnull
from shutil import which
from typing import TYPE_CHECKING, Literal
from packaging.version import Version
from ..settings import settings
def _try_create_chromium_webdriver(scale_factor: float=1) -> WebDriver | None:
    try:
        return create_chromium_webdriver(scale_factor=scale_factor)
    except Exception:
        return None