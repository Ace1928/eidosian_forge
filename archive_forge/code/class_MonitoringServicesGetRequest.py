from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesGetRequest(_messages.Message):
    """A MonitoringServicesGetRequest object.

  Fields:
    name: Required. Resource name of the Service. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
  """
    name = _messages.StringField(1, required=True)