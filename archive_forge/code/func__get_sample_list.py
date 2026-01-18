import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def _get_sample_list(self, samplesize, num_samples, replacement=True):
    samplelist = list()
    scenario_numbers = list(range(len(self.callback_data)))
    if num_samples is None:
        for i, l in enumerate(combinations(scenario_numbers, samplesize)):
            samplelist.append((i, np.sort(l)))
    else:
        for i in range(num_samples):
            attempts = 0
            unique_samples = 0
            duplicate = False
            while unique_samples <= len(self._return_theta_names()) and (not duplicate):
                sample = np.random.choice(scenario_numbers, samplesize, replace=replacement)
                sample = np.sort(sample).tolist()
                unique_samples = len(np.unique(sample))
                if sample in samplelist:
                    duplicate = True
                attempts += 1
                if attempts > num_samples:
                    raise RuntimeError('Internal error: timeout constructing\n                                           a sample, the dim of theta may be too\n                                           close to the samplesize')
            samplelist.append((i, sample))
    return samplelist