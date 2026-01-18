from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsUptimeCheckConfigsPatchRequest(_messages.Message):
    """A MonitoringProjectsUptimeCheckConfigsPatchRequest object.

  Fields:
    name: Identifier. A unique resource name for this Uptime check
      configuration. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]
      [PROJECT_ID_OR_NUMBER] is the Workspace host project associated with the
      Uptime check.This field should be omitted when creating the Uptime check
      configuration; on create, the resource name is assigned by the server
      and included in the response.
    updateMask: Optional. If present, only the listed fields in the current
      Uptime check configuration are updated with values from the new
      configuration. If this field is empty, then the current configuration is
      completely replaced with the new configuration.
    uptimeCheckConfig: A UptimeCheckConfig resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    uptimeCheckConfig = _messages.MessageField('UptimeCheckConfig', 3)