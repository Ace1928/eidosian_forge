from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1StorageFormatJsonOptions(_messages.Message):
    """Describes JSON data format.

  Fields:
    encoding: Optional. The character encoding of the data. Accepts "US-
      ASCII", "UTF-8" and "ISO-8859-1". Defaults to UTF-8 if not specified.
  """
    encoding = _messages.StringField(1)