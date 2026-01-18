import functools
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations
from tensorflow.python.util.tf_export import tf_export
class EagerGraphCombination(test_combinations.TestCombination):
    """Run the test in Graph or Eager mode.

  The optional `mode` parameter controls the test's execution mode.  Its
  accepted values are "graph" or "eager" literals.
  """

    def context_managers(self, kwargs):
        mode = kwargs.pop('mode', None)
        if mode is None:
            return []
        elif mode == 'eager':
            return [context.eager_mode()]
        elif mode == 'graph':
            return [ops.Graph().as_default(), context.graph_mode()]
        else:
            raise ValueError(f"Argument 'mode' must be either 'eager' or 'graph'. Received: {mode}.")

    def parameter_modifiers(self):
        return [test_combinations.OptionalParameter('mode')]