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
def _process_multiple_axis_spanning_shapes(self, shape_args, row, col, shape_type, exclude_empty_subplots=True, annotation=None, **kwargs):
    """
        Add a shape or multiple shapes and call _make_axis_spanning_layout_object on
        all the new shapes.
        """
    if shape_type in ['vline', 'vrect']:
        direction = 'vertical'
    elif shape_type in ['hline', 'hrect']:
        direction = 'horizontal'
    else:
        raise ValueError("Bad shape_type %s, needs to be one of 'vline', 'hline', 'vrect', 'hrect'" % (shape_type,))
    if (row is not None or col is not None) and (not self._has_subplots()):
        row = None
        col = None
    n_shapes_before = len(self.layout['shapes'])
    n_annotations_before = len(self.layout['annotations'])
    shape_kwargs, annotation_kwargs = shapeannotation.split_dict_by_key_prefix(kwargs, 'annotation_')
    augmented_annotation = shapeannotation.axis_spanning_shape_annotation(annotation, shape_type, shape_args, annotation_kwargs)
    self.add_shape(row=row, col=col, exclude_empty_subplots=exclude_empty_subplots, **_combine_dicts([shape_args, shape_kwargs]))
    if augmented_annotation is not None:
        self.add_annotation(augmented_annotation, row=row, col=col, exclude_empty_subplots=exclude_empty_subplots, yref=shape_kwargs.get('yref', 'y'))
    for layout_obj, n_layout_objs_before in zip(['shapes', 'annotations'], [n_shapes_before, n_annotations_before]):
        n_layout_objs_after = len(self.layout[layout_obj])
        if n_layout_objs_after > n_layout_objs_before and (row is None and col is None):
            if self.layout[layout_obj][-1].xref is None:
                self.layout[layout_obj][-1].update(xref='x')
            if self.layout[layout_obj][-1].yref is None:
                self.layout[layout_obj][-1].update(yref='y')
        new_layout_objs = tuple(filter(lambda x: x is not None, [self._make_axis_spanning_layout_object(direction, self.layout[layout_obj][n]) for n in range(n_layout_objs_before, n_layout_objs_after)]))
        self.layout[layout_obj] = self.layout[layout_obj][:n_layout_objs_before] + new_layout_objs