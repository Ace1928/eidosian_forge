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
class LogFilter(logging.Filter):

    def filter(self, record):
        if 'Session ' not in record.msg:
            return True
        session_id = record.args[0]
        if session_id in log_sessions:
            return False
        elif session_id not in session_filter.options:
            session_filter.options = [session_id] + session_filter.options
        return True