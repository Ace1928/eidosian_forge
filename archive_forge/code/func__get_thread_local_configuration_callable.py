import contextlib
import threading
import weakref
from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.util import traceback_utils
def _get_thread_local_configuration_callable():
    if traceback_utils.is_traceback_filtering_enabled():
        thread_local_callables = {traceback_utils.enable_traceback_filtering}
    else:
        thread_local_callables = {traceback_utils.disable_traceback_filtering}
    return thread_local_callables