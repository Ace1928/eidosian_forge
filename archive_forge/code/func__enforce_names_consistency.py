import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _enforce_names_consistency(specs):
    """Enforces that either all specs have names or none do."""

    def _has_name(spec):
        return hasattr(spec, 'name') and spec.name is not None

    def _clear_name(spec):
        spec = copy.deepcopy(spec)
        if hasattr(spec, 'name'):
            spec._name = None
        return spec
    flat_specs = nest.flatten(specs)
    name_inconsistency = any((_has_name(s) for s in flat_specs)) and (not all((_has_name(s) for s in flat_specs)))
    if name_inconsistency:
        specs = nest.map_structure(_clear_name, specs)
    return specs