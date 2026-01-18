import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
def _create_session(self):
    while True:
        try:
            return self._sess_creator.create_session()
        except _PREEMPTION_ERRORS as e:
            logging.info('An error was raised while a session was being created. This may be due to a preemption of a connected worker or parameter server. A new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: %s', e)