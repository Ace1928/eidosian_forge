import numpy as np
import pandas as pd
import graphtools
from . import utils
from . import filter
from graphtools.estimator import GraphEstimator, attribute
from functools import partial
def _reset_filter(self):
    self.filt = None
    self.sample_densities = None