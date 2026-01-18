from __future__ import annotations
import functools
import os
import pathlib
from typing import (
import param
from bokeh.models import ImportedStyleSheet
from bokeh.themes import Theme as _BkTheme, _dark_minimal, built_in_themes
from ..config import config
from ..io.resources import (
from ..util import relative_to
@classmethod
def _patch_modifiers(cls, doc, modifiers, cache):
    if 'stylesheets' in modifiers:
        stylesheets = []
        for sts in modifiers['stylesheets']:
            if sts.endswith('.css'):
                if cache and sts in cache:
                    sts = cache[sts]
                else:
                    sts = ImportedStyleSheet(url=sts)
                    if cache is not None:
                        cache[sts.url] = sts
            stylesheets.append(sts)
        modifiers['stylesheets'] = stylesheets