from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV2alphaApi(_messages.Message):
    """API details for APIs exposed by a Service.

  Fields:
    name: The name of the API.
    operations: The operations this API exposes.
    version: The version of the API.
  """
    name = _messages.StringField(1)
    operations = _messages.MessageField('GoogleApiServiceusageV2alphaOperation', 2, repeated=True)
    version = _messages.StringField(3)