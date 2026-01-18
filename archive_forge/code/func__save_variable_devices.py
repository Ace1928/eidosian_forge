import enum
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _save_variable_devices(self):
    """Checks whether variable devices should be saved."""
    return self != VariablePolicy.NONE