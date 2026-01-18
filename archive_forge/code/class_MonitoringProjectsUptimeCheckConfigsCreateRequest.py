from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsUptimeCheckConfigsCreateRequest(_messages.Message):
    """A MonitoringProjectsUptimeCheckConfigsCreateRequest object.

  Fields:
    parent: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) in which to
      create the Uptime check. The format is: projects/[PROJECT_ID_OR_NUMBER]
    uptimeCheckConfig: A UptimeCheckConfig resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    uptimeCheckConfig = _messages.MessageField('UptimeCheckConfig', 2)