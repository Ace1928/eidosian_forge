from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimulatedResult(_messages.Message):
    """Possible test result.

  Fields:
    error: Error encountered during the test.
    finding: Finding that would be published for the test case, if a violation
      is detected.
    noViolation: Indicates that the test case does not trigger any violation.
  """
    error = _messages.MessageField('Status', 1)
    finding = _messages.MessageField('Finding', 2)
    noViolation = _messages.MessageField('Empty', 3)