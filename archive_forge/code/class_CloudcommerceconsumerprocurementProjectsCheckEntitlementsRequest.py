from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsCheckEntitlementsRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsCheckEntitlementsRequest
  object.

  Fields:
    parent: Required. The consumer project Format: `projects/{project_number}`
      Required.
    service: Required. The one platform service name. Format:
      `services/{service_name}`. Required.
  """
    parent = _messages.StringField(1, required=True)
    service = _messages.StringField(2)