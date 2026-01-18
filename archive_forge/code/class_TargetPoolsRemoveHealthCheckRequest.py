from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetPoolsRemoveHealthCheckRequest(_messages.Message):
    """A TargetPoolsRemoveHealthCheckRequest object.

  Fields:
    healthChecks: Health check URL to be removed. This can be a full or valid
      partial URL. For example, the following are valid URLs: -
      https://www.googleapis.com/compute/beta/projects/project
      /global/httpHealthChecks/health-check -
      projects/project/global/httpHealthChecks/health-check -
      global/httpHealthChecks/health-check
  """
    healthChecks = _messages.MessageField('HealthCheckReference', 1, repeated=True)