from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertChart(_messages.Message):
    """A chart that displays alert policy data.

  Fields:
    name: Required. The resource name of the alert policy. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[ALERT_POLICY_ID]
  """
    name = _messages.StringField(1)