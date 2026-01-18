from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
@param.depends('previous', watch=True)
def _previous(self):
    prev_state, prev_stage = (self._state, self._stage)
    self._stage = self._prev_stage
    try:
        if self._stage in self._states:
            self._state = self._states[self._stage]
            self.stage[0] = self._state.panel()
        else:
            self.stage[0] = self._init_stage()
        self._block = True
    except Exception as e:
        self.error[:] = [self._get_error_button(e)]
        self._error = self._stage
        self._stage = prev_stage
        self._state = prev_state
        if self.debug:
            raise e
    else:
        self.error[:] = []
        self._error = None
        self._update_button()
        self._route.pop()
    finally:
        self._update_progress()