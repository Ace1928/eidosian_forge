from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsAlertPoliciesDeleteRequest(_messages.Message):
    """A MonitoringProjectsAlertPoliciesDeleteRequest object.

  Fields:
    name: Required. The alerting policy to delete. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[ALERT_POLICY_ID] For more
      information, see AlertPolicy.
  """
    name = _messages.StringField(1, required=True)