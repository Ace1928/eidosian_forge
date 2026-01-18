from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollUpValueValuesEnum(_messages.Enum):
    """Rollup test status of multiple steps that were run with the same
    configuration as a group.

    Values:
      unset: Do not use. For proto versioning only.
      success: The test matrix run was successful, for instance: - All the
        test cases passed. - Robo did not detect a crash of the application
        under test.
      failure: A run failed, for instance: - One or more test case failed. - A
        test timed out. - The application under test crashed.
      inconclusive: Something unexpected happened. The run should still be
        considered unsuccessful but this is likely a transient problem and re-
        running the test might be successful.
      skipped: All tests were skipped, for instance: - All device
        configurations were incompatible.
      flaky: A group of steps that were run with the same configuration had
        both failure and success outcomes.
    """
    unset = 0
    success = 1
    failure = 2
    inconclusive = 3
    skipped = 4
    flaky = 5