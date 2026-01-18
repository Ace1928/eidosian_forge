import os
import re
import sys
import uuid
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, Sequence
from copy import copy as ccopy
from copy import deepcopy
import datetime
from html import escape
from typing import (
import numpy as np
import xarray as xr
from packaging import version
from ..rcparams import rcParams
from ..utils import HtmlTemplate, _subset_list, _var_names, either_dict_or_kwargs
from .base import _extend_xr_method, _make_json_serializable, dict_to_dataset
def _wrap_xarray_method(self, method, groups=None, filter_groups=None, inplace=False, args=None, **kwargs):
    """Extend and xarray.Dataset method to InferenceData object.

        Parameters
        ----------
        method: str
            Method to be extended. Must be a ``xarray.Dataset`` method.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        **kwargs: mapping, optional
            Keyword arguments passed to the xarray Dataset method.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        Examples
        --------
        Compute the mean of `posterior_groups`:

        .. ipython::

            In [1]: import arviz as az
               ...: idata = az.load_arviz_data("non_centered_eight")
               ...: idata_means = idata._wrap_xarray_method("mean", groups="latent_vars")
               ...: print(idata_means.posterior)
               ...: print(idata_means.observed_data)

        .. ipython::

            In [1]: idata_stack = idata._wrap_xarray_method(
               ...:     "stack",
               ...:     groups=["posterior_groups", "prior_groups"],
               ...:     sample=["chain", "draw"]
               ...: )
               ...: print(idata_stack.posterior)
               ...: print(idata_stack.prior)
               ...: print(idata_stack.observed_data)

        """
    if args is None:
        args = []
    groups = self._group_names(groups, filter_groups)
    method = getattr(xr.Dataset, method)
    out = self if inplace else deepcopy(self)
    for group in groups:
        dataset = getattr(self, group)
        dataset = method(dataset, *args, **kwargs)
        setattr(out, group, dataset)
    if inplace:
        return None
    else:
        return out