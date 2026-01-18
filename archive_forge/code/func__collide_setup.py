from __future__ import annotations
import typing
from abc import ABC
from copy import copy
from warnings import warn
import numpy as np
from .._utils import check_required_aesthetics, groupby_apply
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import X_AESTHETICS, Y_AESTHETICS
@classmethod
def _collide_setup(cls, data, params):
    xminmax = ['xmin', 'xmax']
    width = params.get('width', None)
    if width is not None:
        if not all((col in data.columns for col in xminmax)):
            data['xmin'] = data['x'] - width / 2
            data['xmax'] = data['x'] + width / 2
    else:
        if not all((col in data.columns for col in xminmax)):
            data['xmin'] = data['x']
            data['xmax'] = data['x']
        widths = (data['xmax'] - data['xmin']).drop_duplicates()
        widths = widths[~np.isnan(widths)]
        width = widths.iloc[0]
    return (data, width)