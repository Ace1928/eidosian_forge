import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
def _to_rgb(data, mesh_data_kind):
    """
    Helper function to convert array to RGB(A) where required
    """
    if mesh_data_kind in ('rgb', 'rgba'):
        cmap = plt.get_cmap()
        norm = mcolors.Normalize()
        new_data = cmap(norm(data))
        if mesh_data_kind == 'rgb':
            new_data = new_data[..., 0:3]
            if np.ma.is_masked(data):
                mask = np.ma.getmaskarray(data)
                mask = np.broadcast_to(mask[..., np.newaxis], new_data.shape).copy()
                new_data = np.ma.array(new_data, mask=mask)
        return new_data
    return data