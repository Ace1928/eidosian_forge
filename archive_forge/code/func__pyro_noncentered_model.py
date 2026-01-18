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
def _pyro_noncentered_model(J, sigma, y=None):
    import pyro
    import pyro.distributions as dist
    mu = pyro.sample('mu', dist.Normal(0, 5))
    tau = pyro.sample('tau', dist.HalfCauchy(5))
    with pyro.plate('J', J):
        eta = pyro.sample('eta', dist.Normal(0, 1))
        theta = mu + tau * eta
        return pyro.sample('obs', dist.Normal(theta, sigma), obs=y)