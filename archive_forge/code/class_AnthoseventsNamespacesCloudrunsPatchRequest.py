from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudrunsPatchRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudrunsPatchRequest object.

  Fields:
    cloudRun: A CloudRun resource to be passed as the request body.
    name: The name of the CloudRun resource being updated.
  """
    cloudRun = _messages.MessageField('CloudRun', 1)
    name = _messages.StringField(2, required=True)