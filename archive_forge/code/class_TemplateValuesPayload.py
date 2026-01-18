from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemplateValuesPayload(_messages.Message):
    """Message for storing a TEMPLATED ConfigType Config resource.

  Fields:
    data: Required. User provided content of a ConfigVersion in reference to a
      TemplateInstance. It can hold references to Secret Manager SecretVersion
      resources & must hold all template variable definitions required by a
      TemplateInstance to be rendered properly.
  """
    data = _messages.BytesField(1)