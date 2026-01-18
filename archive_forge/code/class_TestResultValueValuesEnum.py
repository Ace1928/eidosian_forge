from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestResultValueValuesEnum(_messages.Enum):
    """Whether the test case passed in the agent environment.

    Values:
      TEST_RESULT_UNSPECIFIED: Not specified. Should never be used.
      PASSED: The test passed.
      FAILED: The test did not pass.
    """
    TEST_RESULT_UNSPECIFIED = 0
    PASSED = 1
    FAILED = 2