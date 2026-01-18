import multiprocessing.pool
import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.optimize._minimize
import cirq
import cirq_google as cg
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import (
def _minimize_patch_2(*args, **kwargs):
    return _minimize_patch(*args, **kwargs, x0_should_be=x0_should_be)