from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudStorageResult(_messages.Message):
    """Final results written to Cloud Storage.

  Fields:
    srtFormatUri: The Cloud Storage URI to which recognition results were
      written as SRT formatted captions. This is populated only when `SRT`
      output is requested.
    uri: The Cloud Storage URI to which recognition results were written.
    vttFormatUri: The Cloud Storage URI to which recognition results were
      written as VTT formatted captions. This is populated only when `VTT`
      output is requested.
  """
    srtFormatUri = _messages.StringField(1)
    uri = _messages.StringField(2)
    vttFormatUri = _messages.StringField(3)