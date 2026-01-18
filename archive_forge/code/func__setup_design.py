from __future__ import annotations
import os
import sys
import uuid
from functools import partial
from pathlib import Path, PurePath
from typing import (
import jinja2
import param
from bokeh.document.document import Document
from bokeh.models import LayoutDOM
from bokeh.settings import settings as _settings
from pyviz_comms import JupyterCommManager as _JupyterCommManager
from ..config import _base_config, config, panel_extension
from ..io.document import init_doc
from ..io.model import add_to_doc
from ..io.notebook import render_template
from ..io.notifications import NotificationArea
from ..io.resources import (
from ..io.save import save
from ..io.state import curdoc_locked, state
from ..layout import Column, GridSpec, ListLike
from ..models.comm_manager import CommManager
from ..pane import (
from ..pane.image import ImageBase
from ..reactive import ReactiveHTML
from ..theme.base import (
from ..util import isurl
from ..viewable import (
from ..widgets import Button
from ..widgets.indicators import BooleanIndicator, LoadingSpinner
@param.depends('design', watch=True)
def _setup_design(self):
    self._design = self.design(theme=self.theme)