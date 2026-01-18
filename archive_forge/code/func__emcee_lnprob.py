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
def _emcee_lnprob(theta, y, sigma):
    """Proper function to allow pickling."""
    mu, tau, eta = (theta[0], theta[1], theta[2:])
    prior = _emcee_lnprior(theta)
    like_vect = -((mu + tau * eta - y) / sigma) ** 2
    like = np.sum(like_vect)
    return (like + prior, (like_vect, np.random.normal(mu + tau * eta, sigma)))