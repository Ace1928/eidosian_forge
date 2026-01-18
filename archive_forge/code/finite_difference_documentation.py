from typing import Sequence, Callable
import functools
from functools import partial
from warnings import warn
import numpy as np
from scipy.special import factorial
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .general_shift_rules import generate_shifted_tapes
from .gradient_transform import (
Auxiliary function for post-processing one batch of results corresponding to finite shots or a single
        component of a shot vector