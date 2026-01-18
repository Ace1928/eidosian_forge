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
def _apply_modifiers(cls, viewable: Viewable, mref: str, theme: Theme, isolated: bool, cache={}, document=None) -> None:
    if mref not in viewable._models:
        return
    model, _ = viewable._models[mref]
    modifiers, child_modifiers = cls._get_modifiers(viewable, theme, isolated)
    cls._patch_modifiers(model.document or document, modifiers, cache)
    if child_modifiers:
        for child in viewable:
            cls._apply_params(child, mref, child_modifiers, document)
    if modifiers:
        cls._apply_params(viewable, mref, modifiers, document)