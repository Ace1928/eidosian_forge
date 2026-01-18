import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
class KerasModeCombination(test_combinations.TestCombination):
    """Combination for Keras test mode.

  It by default includes v1_session, v2_eager and v2_tf_function.
  """

    def context_managers(self, kwargs):
        run_eagerly = kwargs.pop('run_eagerly', None)
        if run_eagerly is not None:
            return [testing_utils.run_eagerly_scope(run_eagerly)]
        else:
            return []

    def parameter_modifiers(self):
        return [test_combinations.OptionalParameter('run_eagerly')]