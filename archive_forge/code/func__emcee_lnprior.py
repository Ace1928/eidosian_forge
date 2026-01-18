import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def _emcee_lnprior(theta):
    """Proper function to allow pickling."""
    mu, tau, eta = (theta[0], theta[1], theta[2:])
    if tau < 0:
        return -np.inf
    prior_tau = -np.log(tau ** 2 + 25 ** 2)
    prior_mu = -(mu / 10) ** 2
    prior_eta = -np.sum(eta ** 2)
    return prior_mu + prior_tau + prior_eta