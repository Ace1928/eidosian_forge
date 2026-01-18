from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IncidentList(_messages.Message):
    """A widget that displays a list of incidents

  Fields:
    monitoredResources: Optional. The monitored resource for which incidents
      are listed. The resource doesn't need to be fully specified. That is,
      you can specify the resource type but not the values of the resource
      labels. The resource type and labels are used for filtering.
    policyNames: Optional. A list of alert policy names to filter the incident
      list by. Don't include the project ID prefix in the policy name. For
      example, use alertPolicies/utilization.
  """
    monitoredResources = _messages.MessageField('MonitoredResource', 1, repeated=True)
    policyNames = _messages.StringField(2, repeated=True)