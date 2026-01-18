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
def _perform_batch_animate(self, animation_opts):
    """
        Perform the batch animate operation

        This method should be called with the batch_animate() context
        manager exits.

        Parameters
        ----------
        animation_opts : dict
            Animation options as accepted by frontend Plotly.animation command

        Returns
        -------
        None
        """
    restyle_data, relayout_data, trace_indexes = self._build_update_params_from_batch()
    restyle_changes, relayout_changes, trace_indexes = self._perform_plotly_update(restyle_data, relayout_data, trace_indexes)
    if self._batch_trace_edits:
        animate_styles, animate_trace_indexes = zip(*[(trace_style, trace_index) for trace_index, trace_style in self._batch_trace_edits.items()])
    else:
        animate_styles, animate_trace_indexes = ({}, [])
    animate_layout = copy(self._batch_layout_edits)
    self._send_animate_msg(styles_data=list(animate_styles), relayout_data=animate_layout, trace_indexes=list(animate_trace_indexes), animation_opts=animation_opts)
    self._batch_layout_edits.clear()
    self._batch_trace_edits.clear()
    if restyle_changes:
        self._dispatch_trace_change_callbacks(restyle_changes, trace_indexes)
    if relayout_changes:
        self._dispatch_layout_change_callbacks(relayout_changes)