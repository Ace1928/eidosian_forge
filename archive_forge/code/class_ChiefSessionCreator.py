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
@tf_export(v1=['train.ChiefSessionCreator'])
class ChiefSessionCreator(SessionCreator):
    """Creates a tf.compat.v1.Session for a chief."""

    def __init__(self, scaffold=None, master='', config=None, checkpoint_dir=None, checkpoint_filename_with_path=None):
        """Initializes a chief session creator.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string.  Optional path to a directory where to restore
        variables.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
    """
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_filename_with_path = checkpoint_filename_with_path
        self._scaffold = scaffold or Scaffold()
        self._session_manager = None
        self._master = master
        self._config = config

    def _get_session_manager(self):
        """Gets or creates a SessionManager."""
        if self._session_manager:
            return self._session_manager
        self._session_manager = sm.SessionManager(local_init_op=self._scaffold.local_init_op, local_init_feed_dict=self._scaffold.local_init_feed_dict, ready_op=self._scaffold.ready_op, ready_for_local_init_op=self._scaffold.ready_for_local_init_op, graph=ops.get_default_graph())
        return self._session_manager

    def create_session(self):
        self._scaffold.finalize()
        return self._get_session_manager().prepare_session(self._master, saver=self._scaffold.saver, checkpoint_dir=self._checkpoint_dir, checkpoint_filename_with_path=self._checkpoint_filename_with_path, config=self._config, init_op=self._scaffold.init_op, init_feed_dict=self._scaffold.init_feed_dict, init_fn=self._scaffold.init_fn)