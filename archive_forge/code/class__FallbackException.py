from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
class _FallbackException(Exception):
    """Exception class to handle fallback from the fastpath.

  The fastpath that we refer to here is the one implemented to reduce per-op
  overheads (TFE_Py_FastPathExecute_C). If the conditions for executing the op
  on the fastpath are not met, we fallback to a safer (and more complete)
  slowpath, and this Exception is raised to signal that transition.
  """
    pass