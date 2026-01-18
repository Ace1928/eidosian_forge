import datetime as dt
import logging
import os
import sys
import time
from functools import partial
import bokeh
import numpy as np
import pandas as pd
import param
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from ..config import config, panel_extension as extension
from ..depends import bind
from ..layout import (
from ..pane import HTML, Bokeh
from ..template import FastListTemplate
from ..widgets import (
from ..widgets.indicators import Trend
from .logging import (
from .notebook import push_notebook
from .profile import profiling_tabs
from .server import set_curdoc
from .state import state
class _LogTabulator(Tabulator):
    _update_defaults = {'theme': 'midnight', 'layout': 'fit_data_stretch', 'show_index': False, 'sorters': [{'field': 'datetime', 'dir': 'dsc'}], 'disabled': True, 'pagination': 'local', 'page_size': 18}

    def __init__(self, **params):
        params['value'] = self._create_frame()
        params = {**self._update_defaults, **params}
        super().__init__(**params)

    @staticmethod
    def _create_frame(data=None):
        columns = ['datetime', 'level', 'app', 'session', 'message']
        if data is None:
            return pd.DataFrame(columns=columns)
        else:
            return pd.Series(data, index=columns)

    def write(self, log):
        try:
            s = log.strip().split(' ')
            datetime = f'{s[0]} {s[1]}'
            level = s[2][:-1]
            app = s[3]
            session = int(s[6])
            message = ' '.join(s[7:])
            df = self._create_frame([datetime, level, app, session, message])
            self.stream(df, follow=False)
        except Exception:
            pass