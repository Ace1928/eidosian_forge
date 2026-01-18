from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetPoolsAddHealthCheckRequest(_messages.Message):
    """A TargetPoolsAddHealthCheckRequest object.

  Fields:
    healthChecks: The HttpHealthCheck to add to the target pool.
  """
    healthChecks = _messages.MessageField('HealthCheckReference', 1, repeated=True)