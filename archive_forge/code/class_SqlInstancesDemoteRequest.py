from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesDemoteRequest(_messages.Message):
    """A SqlInstancesDemoteRequest object.

  Fields:
    instance: Required. The name of the Cloud SQL instance.
    instancesDemoteRequest: A InstancesDemoteRequest resource to be passed as
      the request body.
    project: Required. The project ID of the project that contains the
      instance.
  """
    instance = _messages.StringField(1, required=True)
    instancesDemoteRequest = _messages.MessageField('InstancesDemoteRequest', 2)
    project = _messages.StringField(3, required=True)