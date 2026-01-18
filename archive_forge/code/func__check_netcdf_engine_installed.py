from __future__ import annotations
import os
import pathlib
from typing import TYPE_CHECKING
import numpy as np
from xarray.backends.api import open_dataset as _open_dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
def _check_netcdf_engine_installed(name):
    version = file_formats.get(name)
    if version == 3:
        try:
            import scipy
        except ImportError:
            try:
                import netCDF4
            except ImportError:
                raise ImportError(f'opening tutorial dataset {name} requires either scipy or netCDF4 to be installed.')
    if version == 4:
        try:
            import h5netcdf
        except ImportError:
            try:
                import netCDF4
            except ImportError:
                raise ImportError(f'opening tutorial dataset {name} requires either h5netcdf or netCDF4 to be installed.')