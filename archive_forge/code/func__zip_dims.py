from itertools import product, tee
import numpy as np
import xarray as xr
from .labels import BaseLabeller
def _zip_dims(new_dims, vals):
    return [dict(zip(new_dims, prod)) for prod in product(*vals)]