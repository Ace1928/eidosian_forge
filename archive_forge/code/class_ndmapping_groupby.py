import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
class ndmapping_groupby(param.ParameterizedFunction):
    """
    Apply a groupby operation to an NdMapping, using pandas to improve
    performance (if available).
    """
    sort = param.Boolean(default=False, doc='Whether to apply a sorted groupby')

    def __call__(self, ndmapping, dimensions, container_type, group_type, sort=False, **kwargs):
        return self.groupby_pandas(ndmapping, dimensions, container_type, group_type, sort=sort, **kwargs)

    @param.parameterized.bothmethod
    def groupby_pandas(self_or_cls, ndmapping, dimensions, container_type, group_type, sort=False, **kwargs):
        if 'kdims' in kwargs:
            idims = [ndmapping.get_dimension(d) for d in kwargs['kdims']]
        else:
            idims = [dim for dim in ndmapping.kdims if dim not in dimensions]
        all_dims = [d.name for d in ndmapping.kdims]
        inds = [ndmapping.get_dimension_index(dim) for dim in idims]
        getter = operator.itemgetter(*inds) if inds else lambda x: ()
        multi_index = pd.MultiIndex.from_tuples(ndmapping.keys(), names=all_dims)
        df = pd.DataFrame(list(map(wrap_tuple, ndmapping.values())), index=multi_index)
        kwargs = dict(dict(get_param_values(ndmapping), kdims=idims), sort=sort, **kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, message='Creating a Groupby object with a length-1')
            groups = ((wrap_tuple(k), group_type(dict(unpack_group(group, getter)), **kwargs)) for k, group in df.groupby(level=[d.name for d in dimensions], sort=sort))
        if sort:
            selects = list(get_unique_keys(ndmapping, dimensions))
            groups = sorted(groups, key=lambda x: selects.index(x[0]))
        return container_type(groups, kdims=dimensions, sort=sort)

    @param.parameterized.bothmethod
    def groupby_python(self_or_cls, ndmapping, dimensions, container_type, group_type, sort=False, **kwargs):
        idims = [dim for dim in ndmapping.kdims if dim not in dimensions]
        dim_names = [dim.name for dim in dimensions]
        selects = get_unique_keys(ndmapping, dimensions)
        selects = group_select(list(selects))
        groups = [(k, group_type(v.reindex(idims) if hasattr(v, 'kdims') else [((), v)], **kwargs)) for k, v in iterative_select(ndmapping, dim_names, selects)]
        return container_type(groups, kdims=dimensions)