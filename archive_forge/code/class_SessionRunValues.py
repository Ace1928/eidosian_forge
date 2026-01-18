import collections
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.SessionRunValues'])
class SessionRunValues(collections.namedtuple('SessionRunValues', ['results', 'options', 'run_metadata'])):
    """Contains the results of `Session.run()`.

  In the future we may use this object to add more information about result of
  run without changing the Hook API.

  Args:
    results: The return values from `Session.run()` corresponding to the fetches
      attribute returned in the RunArgs. Note that this has the same shape as
      the RunArgs fetches.  For example:
        fetches = global_step_tensor
        => results = nparray(int)
        fetches = [train_op, summary_op, global_step_tensor]
        => results = [None, nparray(string), nparray(int)]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
        => results = {'step': nparray(int), 'summ': nparray(string)}
    options: `RunOptions` from the `Session.run()` call.
    run_metadata: `RunMetadata` from the `Session.run()` call.
  """