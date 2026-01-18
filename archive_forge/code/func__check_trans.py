from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import censor, expand_range_distinct, rescale, zero_range
from .._utils import match
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import range_view, scale_view
from ._expand import expand_range
from .range import RangeContinuous
from .scale import scale
def _check_trans(self, t):
    """
        Check if transform t is valid

        When scales specialise on a specific transform (other than
        the identity transform), the user should know when they
        try to change the transform.

        Parameters
        ----------
        t : mizani.transforms.trans
            Transform object
        """
    orig_trans_name = self.__class__._trans
    new_trans_name = t.__class__.__name__
    if new_trans_name.endswith('_trans'):
        new_trans_name = new_trans_name[:-6]
    if orig_trans_name not in ('identity', new_trans_name):
        warn('You have changed the transform of a specialised scale. The result may not be what you expect.\nOriginal transform: {}\nNew transform: {}'.format(orig_trans_name, new_trans_name), PlotnineWarning, stacklevel=2)