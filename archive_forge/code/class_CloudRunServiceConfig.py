from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunServiceConfig(_messages.Message):
    """Message for Cloud Run service configs.

  Fields:
    config: Configuration for the service.
    image: The container image to deploy the service with.
    resources: Bindings to other resources.
  """
    config = _messages.MessageField('ServiceSettingsConfig', 1)
    image = _messages.StringField(2)
    resources = _messages.MessageField('ServiceResourceBindingConfig', 3, repeated=True)