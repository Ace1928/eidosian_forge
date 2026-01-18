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
class EightSchools:

    @bm.random_variable
    def mu(self):
        return dist.Normal(0, 5)

    @bm.random_variable
    def tau(self):
        return dist.HalfCauchy(5)

    @bm.random_variable
    def eta(self):
        return dist.Normal(0, 1).expand((data['J'],))

    @bm.functional
    def theta(self):
        return self.mu() + self.tau() * self.eta()

    @bm.random_variable
    def obs(self):
        return dist.Normal(self.theta(), torch.from_numpy(data['sigma']).float())