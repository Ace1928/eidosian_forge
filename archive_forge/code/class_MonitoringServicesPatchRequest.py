from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesPatchRequest(_messages.Message):
    """A MonitoringServicesPatchRequest object.

  Fields:
    name: Resource name for this Service. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
    service: A Service resource to be passed as the request body.
    updateMask: A set of field paths defining which fields to use for the
      update.
  """
    name = _messages.StringField(1, required=True)
    service = _messages.MessageField('Service', 2)
    updateMask = _messages.StringField(3)