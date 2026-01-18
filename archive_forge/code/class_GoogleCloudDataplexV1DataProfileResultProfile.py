from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileResultProfile(_messages.Message):
    """Contains name, type, mode and field type specific profile information.

  Fields:
    fields: List of fields with structural and profile information for each
      field.
  """
    fields = _messages.MessageField('GoogleCloudDataplexV1DataProfileResultProfileField', 1, repeated=True)