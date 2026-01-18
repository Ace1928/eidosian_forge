from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _CustomSaver(saver_lib.Saver):
    """`Saver` with a different default `latest_filename`.

  This is used in the `CheckpointInputPipelineHook` to avoid conflicts with
  the model ckpt saved by the `CheckpointSaverHook`.
  """

    def __init__(self, var_list, latest_filename, sharded=False):
        super(_CustomSaver, self).__init__(var_list, sharded=sharded)
        self._latest_filename = latest_filename

    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True, strip_default_attrs=False):
        return super(_CustomSaver, self).save(sess, save_path, global_step, latest_filename or self._latest_filename, meta_graph_suffix, write_meta_graph, write_state, strip_default_attrs)