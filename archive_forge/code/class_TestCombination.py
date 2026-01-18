from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.test.combinations.TestCombination', v1=[])
class TestCombination:
    """Customize the behavior of `generate()` and the tests that it executes.

  Here is sequence of steps for executing a test combination:
    1. The test combination is evaluated for whether it should be executed in
       the given environment by calling `should_execute_combination`.
    2. If the test combination is going to be executed, then the arguments for
       all combined parameters are validated.  Some arguments can be handled in
       a special way.  This is achieved by implementing that logic in
       `ParameterModifier` instances that returned from `parameter_modifiers`.
    3. Before executing the test, `context_managers` are installed
       around it.
  """

    def should_execute_combination(self, kwargs):
        """Indicates whether the combination of test arguments should be executed.

    If the environment doesn't satisfy the dependencies of the test
    combination, then it can be skipped.

    Args:
      kwargs:  Arguments that are passed to the test combination.

    Returns:
      A tuple boolean and an optional string.  The boolean False indicates
    that the test should be skipped.  The string would indicate a textual
    description of the reason.  If the test is going to be executed, then
    this method returns `None` instead of the string.
    """
        del kwargs
        return (True, None)

    def parameter_modifiers(self):
        """Returns `ParameterModifier` instances that customize the arguments."""
        return []

    def context_managers(self, kwargs):
        """Return context managers for running the test combination.

    The test combination will run under all context managers that all
    `TestCombination` instances return.

    Args:
      kwargs:  Arguments and their values that are passed to the test
        combination.

    Returns:
      A list of instantiated context managers.
    """
        del kwargs
        return []