from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoredProject(_messages.Message):
    """A project being monitored
  (https://cloud.google.com/monitoring/settings/multiple-projects#create-
  multi) by a Metrics Scope.

  Fields:
    createTime: Output only. The time when this MonitoredProject was created.
    name: Immutable. The resource name of the MonitoredProject. On input, the
      resource name includes the scoping project ID and monitored project ID.
      On output, it contains the equivalent project numbers. Example: location
      s/global/metricsScopes/{SCOPING_PROJECT_ID_OR_NUMBER}/projects/{MONITORE
      D_PROJECT_ID_OR_NUMBER}
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)