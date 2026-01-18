import typing as tp
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import xarray
from .inference_data import InferenceData
from ..rcparams import rcParams
from .base import dict_to_dataset
def _pyjags_samples_to_xarray(self, pyjags_samples: tp.Mapping[str, np.ndarray]) -> tp.Tuple[xarray.Dataset, xarray.Dataset]:
    data, data_warmup = get_draws(pyjags_samples=pyjags_samples, warmup_iterations=self.warmup_iterations, warmup=self.save_warmup)
    return (dict_to_dataset(data, library=self.pyjags, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.pyjags, coords=self.coords, dims=self.dims))