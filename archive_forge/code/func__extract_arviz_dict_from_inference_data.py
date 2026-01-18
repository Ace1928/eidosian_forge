import typing as tp
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import xarray
from .inference_data import InferenceData
from ..rcparams import rcParams
from .base import dict_to_dataset
def _extract_arviz_dict_from_inference_data(idata) -> tp.Mapping[str, np.ndarray]:
    """
    Extract the samples dictionary from an ArviZ inference data object.

    Extracts a dictionary mapping parameter names to NumPy arrays of samples
    with shape (number_of_chains, chain_length, parameter_dimension) from an
    ArviZ inference data object.

    Parameters
    ----------
    idata: InferenceData

    Returns
    -------
    a dictionary mapping variable names to NumPy arrays with shape
             (number_of_chains, chain_length, parameter_dimension)

    """
    variable_name_to_samples_map = {key: np.array(value['data']) for key, value in idata.posterior.to_dict()['data_vars'].items()}
    return variable_name_to_samples_map