from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
def add_debugger(self, debugger):
    """
        Add a debugger to this logging filter.

        Parameters
        ----------
        widg : panel.widgets.Debugger
            The widget displaying the logs.

        Returns
        -------
        None.
        """
    self.debugger = debugger