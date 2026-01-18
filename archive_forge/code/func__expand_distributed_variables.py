import enum
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _expand_distributed_variables(self):
    """Checks whether distributed variables should be expanded."""
    return self == VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES