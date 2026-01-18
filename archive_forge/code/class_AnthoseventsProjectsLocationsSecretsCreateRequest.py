from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsProjectsLocationsSecretsCreateRequest(_messages.Message):
    """A AnthoseventsProjectsLocationsSecretsCreateRequest object.

  Fields:
    parent: Required. The project ID or project number in which this secret
      should be created.
    secret: A Secret resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    secret = _messages.MessageField('Secret', 2)