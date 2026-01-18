import warnings
import pandas
from packaging import version
import os
from modin.config import Parameter
from modin.pandas import arrays, errors
from modin.utils import show_versions
from .. import __version__
from .dataframe import DataFrame
from .general import (
from .io import (
from .plotting import Plotting as plotting
from .series import Series
def _update_engine(publisher: Parameter):
    from modin.config import CpuCount, Engine, IsExperimental, StorageFormat, ValueSource
    os.environ['OMP_NUM_THREADS'] = '1'
    sfmt = StorageFormat.get()
    if sfmt == 'Hdk':
        is_hdk = True
    elif sfmt == 'Omnisci':
        is_hdk = True
        StorageFormat.put('Hdk')
        warnings.warn('The OmniSci storage format has been deprecated. Please use ' + '`StorageFormat.put("hdk")` or `MODIN_STORAGE_FORMAT="hdk"` instead.')
    else:
        is_hdk = False
    if is_hdk and publisher.get_value_source() == ValueSource.DEFAULT:
        publisher.put('Native')
        IsExperimental.put(True)
    if publisher.get() == 'Native' and StorageFormat.get_value_source() == ValueSource.DEFAULT:
        is_hdk = True
        StorageFormat.put('Hdk')
        IsExperimental.put(True)
    if publisher.get() == 'Ray':
        if _is_first_update.get('Ray', True):
            from modin.core.execution.ray.common import initialize_ray
            initialize_ray()
    elif publisher.get() == 'Native':
        if is_hdk:
            os.environ['OMP_NUM_THREADS'] = str(CpuCount.get())
        else:
            raise ValueError(f"Storage format should be 'Hdk' with 'Native' engine, but provided {sfmt}.")
    elif publisher.get() == 'Dask':
        if _is_first_update.get('Dask', True):
            from modin.core.execution.dask.common import initialize_dask
            initialize_dask()
    elif publisher.get() == 'Unidist':
        if _is_first_update.get('Unidist', True):
            from modin.core.execution.unidist.common import initialize_unidist
            initialize_unidist()
    elif publisher.get() not in Engine.NOINIT_ENGINES:
        raise ImportError('Unrecognized execution engine: {}.'.format(publisher.get()))
    _is_first_update[publisher.get()] = False