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
def _handle_qgrid_msg(self, widget, content, buffers=None):
    try:
        self._handle_qgrid_msg_helper(content)
    except Exception as e:
        self.log.error(e)
        self.log.exception('Unhandled exception while handling msg')