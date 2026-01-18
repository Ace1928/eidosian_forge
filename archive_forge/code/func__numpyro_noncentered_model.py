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
def _numpyro_noncentered_model(J, sigma, y=None):
    import numpyro
    import numpyro.distributions as dist
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        eta = numpyro.sample('eta', dist.Normal(0, 1))
        theta = mu + tau * eta
        return numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)