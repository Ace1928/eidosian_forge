from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models.ui import UIElement
from ..util.browser import NEW_PARAM, get_browser_controller
from .notebook import DEFAULT_JUPYTER_URL, ProxyUrlFunc, run_notebook_hook
from .saving import save
from .state import curstate
def _show_file_with_state(obj: UIElement, state: State, new: BrowserTarget, controller: BrowserLike) -> None:
    """

    """
    filename = save(obj, state=state)
    controller.open('file://' + filename, new=NEW_PARAM[new])