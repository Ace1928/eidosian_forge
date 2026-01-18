import functools
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations
from tensorflow.python.util.tf_export import tf_export
class TFVersionCombination(test_combinations.TestCombination):
    """Control the execution of the test in TF1.x and TF2.

  If TF2 is enabled then a test with TF1 test is going to be skipped and vice
  versa.

  Test targets continuously run in TF2 thanks to the tensorflow.v2 TAP target.
  A test can be run in TF2 with bazel by passing --test_env=TF2_BEHAVIOR=1.
  """

    def should_execute_combination(self, kwargs):
        tf_api_version = kwargs.pop('tf_api_version', None)
        if tf_api_version == 1 and tf2.enabled():
            return (False, 'Skipping a TF1.x test when TF2 is enabled.')
        elif tf_api_version == 2 and (not tf2.enabled()):
            return (False, 'Skipping a TF2 test when TF2 is not enabled.')
        return (True, None)

    def parameter_modifiers(self):
        return [test_combinations.OptionalParameter('tf_api_version')]