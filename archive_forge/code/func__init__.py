import warnings
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.diagnostic_gen import (
from statsmodels.discrete._diagnostics_count import (
def _init__(self, results):
    self.results = results