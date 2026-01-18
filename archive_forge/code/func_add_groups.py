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
def add_groups(self, group_dict=None, coords=None, dims=None, **kwargs):
    """Add new groups to InferenceData object.

        Parameters
        ----------
        group_dict : dict of {str : dict or xarray.Dataset}, optional
            Groups to be added
        coords : dict of {str : array_like}, optional
            Coordinates for the dataset
        dims : dict of {str : list of str}, optional
            Dimensions of each variable. The keys are variable names, values are lists of
            coordinates.
        kwargs : dict, optional
            The keyword arguments form of group_dict. One of group_dict or kwargs must be provided.

        Examples
        --------
        Add a ``log_likelihood`` group to the "rugby" example InferenceData after loading.

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")
            del idata.log_likelihood
            idata2 = idata.copy()
            post = idata.posterior
            obs = idata.observed_data
            idata

        Knowing the model, we can compute it manually. In this case however,
        we will generate random samples with the right shape.

        .. jupyter-execute::

            import numpy as np
            rng = np.random.default_rng(73)
            ary = rng.normal(size=(post.sizes["chain"], post.sizes["draw"], obs.sizes["match"]))
            idata.add_groups(
                log_likelihood={"home_points": ary},
                dims={"home_points": ["match"]},
            )
            idata

        This is fine if we have raw data, but a bit inconvenient if we start with labeled
        data already. Why provide dims and coords manually again?
        Let's generate a fake log likelihood (doesn't match the model but it serves just
        the same for illustration purposes here) working from the posterior and
        observed_data groups manually:

        .. jupyter-execute::

            import xarray as xr
            from xarray_einstats.stats import XrDiscreteRV
            from scipy.stats import poisson
            dist = XrDiscreteRV(poisson)
            log_lik = xr.Dataset()
            log_lik["home_points"] = dist.logpmf(obs["home_points"], np.exp(post["atts"]))
            idata2.add_groups({"log_likelihood": log_lik})
            idata2

        Note that in the first example we have used the ``kwargs`` argument
        and in the second we have used the ``group_dict`` one.

        See Also
        --------
        extend : Extend InferenceData with groups from another InferenceData.
        concat : Concatenate InferenceData objects.
        """
    group_dict = either_dict_or_kwargs(group_dict, kwargs, 'add_groups')
    if not group_dict:
        raise ValueError('One of group_dict or kwargs must be provided.')
    repeated_groups = [group for group in group_dict.keys() if group in self._groups]
    if repeated_groups:
        raise ValueError(f'{repeated_groups} group(s) already exists.')
    for group, dataset in group_dict.items():
        if group not in SUPPORTED_GROUPS_ALL:
            warnings.warn(f'The group {group} is not defined in the InferenceData scheme', UserWarning)
        if dataset is None:
            continue
        elif isinstance(dataset, dict):
            if group in ('observed_data', 'constant_data', 'predictions_constant_data') or group not in SUPPORTED_GROUPS_ALL:
                warnings.warn("the default dims 'chain' and 'draw' will be added automatically", UserWarning)
            dataset = dict_to_dataset(dataset, coords=coords, dims=dims)
        elif isinstance(dataset, xr.DataArray):
            if dataset.name is None:
                dataset.name = 'x'
            dataset = dataset.to_dataset()
        elif not isinstance(dataset, xr.Dataset):
            raise ValueError("Arguments to add_groups() must be xr.Dataset, xr.Dataarray or dicts                    (argument '{}' was type '{}')".format(group, type(dataset)))
        if dataset:
            setattr(self, group, dataset)
            if group.startswith(WARMUP_TAG):
                supported_order = [key for key in SUPPORTED_GROUPS_ALL if key in self._groups_warmup]
                if supported_order == self._groups_warmup and group in SUPPORTED_GROUPS_ALL:
                    group_order = [key for key in SUPPORTED_GROUPS_ALL if key in self._groups_warmup + [group]]
                    group_idx = group_order.index(group)
                    self._groups_warmup.insert(group_idx, group)
                else:
                    self._groups_warmup.append(group)
            else:
                supported_order = [key for key in SUPPORTED_GROUPS_ALL if key in self._groups]
                if supported_order == self._groups and group in SUPPORTED_GROUPS_ALL:
                    group_order = [key for key in SUPPORTED_GROUPS_ALL if key in self._groups + [group]]
                    group_idx = group_order.index(group)
                    self._groups.insert(group_idx, group)
                else:
                    self._groups.append(group)