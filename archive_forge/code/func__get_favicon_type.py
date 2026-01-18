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
@staticmethod
def _get_favicon_type(favicon) -> str:
    if not favicon:
        return ''
    elif favicon.endswith('.png'):
        return 'image/png'
    elif favicon.endswith('jpg'):
        return 'image/jpg'
    elif favicon.endswith('gif'):
        return 'image/gif'
    elif favicon.endswith('svg'):
        return 'image/svg'
    elif favicon.endswith('ico'):
        return 'image/x-icon'
    else:
        raise ValueError('favicon type not supported.')