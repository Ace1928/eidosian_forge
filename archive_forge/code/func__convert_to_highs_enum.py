import inspect
import numpy as np
from ._optimize import OptimizeWarning, OptimizeResult
from warnings import warn
from ._highs._highs_wrapper import _highs_wrapper
from ._highs._highs_constants import (
from scipy.sparse import csc_matrix, vstack, issparse
def _convert_to_highs_enum(option, option_str, choices):
    try:
        return choices[option.lower()]
    except AttributeError:
        return choices[option]
    except KeyError:
        sig = inspect.signature(_linprog_highs)
        default_str = sig.parameters[option_str].default
        warn(f'Option {option_str} is {option}, but only values in {set(choices.keys())} are allowed. Using default: {default_str}.', OptimizeWarning, stacklevel=3)
        return choices[default_str]