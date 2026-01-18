import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def _change_selection(self, rows, source, send_msg_to_js=False):
    old_selection = self._selected_rows
    self._selected_rows = rows
    if old_selection == self._selected_rows:
        return
    if send_msg_to_js:
        data_to_send = {'type': 'change_selection', 'rows': rows}
        self.send(data_to_send)
    self._notify_listeners({'name': 'selection_changed', 'old': old_selection, 'new': self._selected_rows, 'source': source})