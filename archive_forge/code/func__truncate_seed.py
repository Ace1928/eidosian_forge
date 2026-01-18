import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _truncate_seed(seed):
    return seed % _MAXINT32