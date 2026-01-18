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
def _apply_hooks(self, viewable: Viewable, root: Model, changed: Viewable, old_models=None) -> None:
    from ..io.state import state
    if root.document in state._stylesheets:
        cache = state._stylesheets[root.document]
    else:
        state._stylesheets[root.document] = cache = {}
    with root.document.models.freeze():
        self._reapply(changed, root, old_models, isolated=False, cache=cache, document=root.document)