import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def _initialize_layout_template(self):
    import plotly.io as pio
    if self._layout_obj._props.get('template', None) is None:
        if pio.templates.default is not None:
            if self._allow_disable_validation:
                self._layout_obj._validate = False
            try:
                if isinstance(pio.templates.default, BasePlotlyType):
                    template_object = pio.templates.default
                else:
                    template_object = pio.templates[pio.templates.default]
                self._layout_obj.template = template_object
            finally:
                self._layout_obj._validate = self._validate