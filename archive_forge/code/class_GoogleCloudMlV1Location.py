from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Location(_messages.Message):
    """A GoogleCloudMlV1Location object.

  Fields:
    capabilities: Capabilities available in the location.
    name: A string attribute.
  """
    capabilities = _messages.MessageField('GoogleCloudMlV1Capability', 1, repeated=True)
    name = _messages.StringField(2)