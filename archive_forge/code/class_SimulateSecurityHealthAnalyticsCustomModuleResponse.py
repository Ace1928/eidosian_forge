from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimulateSecurityHealthAnalyticsCustomModuleResponse(_messages.Message):
    """Response message for simulating a `SecurityHealthAnalyticsCustomModule`
  against a given resource.

  Fields:
    result: Result for test case in the corresponding request.
  """
    result = _messages.MessageField('SimulatedResult', 1)