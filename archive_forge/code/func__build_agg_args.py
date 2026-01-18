from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def _build_agg_args(spec):
    """
    Create transformation functions for a normalized aggregate spec.

    Parameters
    ----------
    spec: a list of (result-column, aggregation-function, input-column) triples.
        To work with all argument forms understood by pandas use
        ``_normalize_spec`` to normalize the argument before passing it on to
        ``_build_agg_args``.

    Returns
    -------
    chunk_funcs: a list of (intermediate-column, function, keyword) triples
        that are applied on grouped chunks of the initial dataframe.

    agg_funcs: a list of (intermediate-column, functions, keyword) triples that
        are applied on the grouped concatenation of the preprocessed chunks.

    finalizers: a list of (result-column, function, keyword) triples that are
        applied after the ``agg_funcs``. They are used to create final results
        from intermediate representations.
    """
    known_np_funcs = {np.min: 'min', np.max: 'max', np.median: 'median', np.std: 'std', np.var: 'var'}
    by_name = {}
    for _, func, input_column in spec:
        key = (funcname(known_np_funcs.get(func, func)), input_column)
        by_name.setdefault(key, []).append((func, input_column))
    for funcs in by_name.values():
        if len(funcs) != 1:
            raise ValueError(f'conflicting aggregation functions: {funcs}')
    chunks = {}
    aggs = {}
    finalizers = []
    for result_column, func, input_column in spec:
        func_args = ()
        func_kwargs = {}
        if isinstance(func, partial):
            func_args, func_kwargs = (func.args, func.keywords)
        if not isinstance(func, Aggregation):
            func = funcname(known_np_funcs.get(func, func))
        impls = _build_agg_args_single(result_column, func, func_args, func_kwargs, input_column)
        for spec in impls['chunk_funcs']:
            chunks[spec[0]] = spec
        for spec in impls['aggregate_funcs']:
            aggs[spec[0]] = spec
        finalizers.append(impls['finalizer'])
    chunks = sorted(chunks.values())
    aggs = sorted(aggs.values())
    return (chunks, aggs, finalizers)