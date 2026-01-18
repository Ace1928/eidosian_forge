from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricsScope(_messages.Message):
    """Represents a Metrics Scope
  (https://cloud.google.com/monitoring/settings#concept-scope) in Cloud
  Monitoring, which specifies one or more Google projects and zero or more AWS
  accounts to monitor together.

  Fields:
    createTime: Output only. The time when this Metrics Scope was created.
    monitoredProjects: Output only. The list of projects monitored by this
      Metrics Scope.
    name: Immutable. The resource name of the Monitoring Metrics Scope. On
      input, the resource name can be specified with the scoping project ID or
      number. On output, the resource name is specified with the scoping
      project number. Example:
      locations/global/metricsScopes/{SCOPING_PROJECT_ID_OR_NUMBER}
    updateTime: Output only. The time when this Metrics Scope record was last
      updated.
  """
    createTime = _messages.StringField(1)
    monitoredProjects = _messages.MessageField('MonitoredProject', 2, repeated=True)
    name = _messages.StringField(3)
    updateTime = _messages.StringField(4)