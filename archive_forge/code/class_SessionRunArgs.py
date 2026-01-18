import collections
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.SessionRunArgs'])
class SessionRunArgs(collections.namedtuple('SessionRunArgs', ['fetches', 'feed_dict', 'options'])):
    """Represents arguments to be added to a `Session.run()` call.

  Args:
    fetches: Exactly like the 'fetches' argument to Session.Run().
      Can be a single tensor or op, a list of 'fetches' or a dictionary
      of fetches.  For example:
        fetches = global_step_tensor
        fetches = [train_op, summary_op, global_step_tensor]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
      Note that this can recurse as expected:
        fetches = {'step': global_step_tensor,
                   'ops': [train_op, check_nan_op]}
    feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
    options: Exactly like the `options` argument to `Session.run()`, i.e., a
      config_pb2.RunOptions proto.
  """

    def __new__(cls, fetches, feed_dict=None, options=None):
        return super(SessionRunArgs, cls).__new__(cls, fetches, feed_dict, options)