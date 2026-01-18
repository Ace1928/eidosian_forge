from __future__ import annotations
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, utils
from xarray.core.alignment import align, reindex_variables
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import Index, PandasIndex
from xarray.core.merge import (
from xarray.core.types import T_DataArray, T_Dataset, T_Variable
from xarray.core.variable import Variable
from xarray.core.variable import concat as concat_vars
def _calc_concat_over(datasets, dim, dim_names, data_vars: T_DataVars, coords, compat):
    """
    Determine which dataset variables need to be concatenated in the result,
    """
    concat_over = set()
    equals = {}
    if dim in dim_names:
        concat_over_existing_dim = True
        concat_over.add(dim)
    else:
        concat_over_existing_dim = False
    concat_dim_lengths = []
    for ds in datasets:
        if concat_over_existing_dim:
            if dim not in ds.dims:
                if dim in ds:
                    ds = ds.set_coords(dim)
        concat_over.update((k for k, v in ds.variables.items() if dim in v.dims))
        concat_dim_lengths.append(ds.sizes.get(dim, 1))

    def process_subset_opt(opt, subset):
        if isinstance(opt, str):
            if opt == 'different':
                if compat == 'override':
                    raise ValueError(f"Cannot specify both {subset}='different' and compat='override'.")
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
                        equals[k] = None
                        variables = [ds.variables[k] for ds in datasets if k in ds.variables]
                        if len(variables) == 1:
                            break
                        elif len(variables) != len(datasets) and opt == 'different':
                            raise ValueError(f"{k!r} not present in all datasets and coords='different'. Either add {k!r} to datasets where it is missing or specify coords='minimal'.")
                        for var in variables[1:]:
                            equals[k] = getattr(variables[0], compat)(var, equiv=lazy_array_equiv)
                            if equals[k] is not True:
                                break
                        if equals[k] is False:
                            concat_over.add(k)
                        elif equals[k] is None:
                            v_lhs = datasets[0].variables[k].load()
                            computed = []
                            for ds_rhs in datasets[1:]:
                                v_rhs = ds_rhs.variables[k].compute()
                                computed.append(v_rhs)
                                if not getattr(v_lhs, compat)(v_rhs):
                                    concat_over.add(k)
                                    equals[k] = False
                                    for ds, v in zip(datasets[1:], computed):
                                        ds.variables[k].data = v.data
                                    break
                            else:
                                equals[k] = True
            elif opt == 'all':
                concat_over.update(set().union(*list((set(getattr(d, subset)) - set(d.dims) for d in datasets))))
            elif opt == 'minimal':
                pass
            else:
                raise ValueError(f'unexpected value for {subset}: {opt}')
        else:
            valid_vars = tuple(getattr(datasets[0], subset))
            invalid_vars = [k for k in opt if k not in valid_vars]
            if invalid_vars:
                if subset == 'coords':
                    raise ValueError(f'the variables {invalid_vars} in coords are not found in the coordinates of the first dataset {valid_vars}')
                else:
                    raise ValueError(f'the variables {invalid_vars} in data_vars are not found in the data variables of the first dataset')
            concat_over.update(opt)
    process_subset_opt(data_vars, 'data_vars')
    process_subset_opt(coords, 'coords')
    return (concat_over, equals, concat_dim_lengths)