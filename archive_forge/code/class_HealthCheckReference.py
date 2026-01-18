from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthCheckReference(_messages.Message):
    """A full or valid partial URL to a health check. For example, the
  following are valid URLs: -
  https://www.googleapis.com/compute/beta/projects/project-
  id/global/httpHealthChecks/health-check - projects/project-
  id/global/httpHealthChecks/health-check - global/httpHealthChecks/health-
  check

  Fields:
    healthCheck: A string attribute.
  """
    healthCheck = _messages.StringField(1)