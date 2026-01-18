from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1GcsSource(_messages.Message):
    """The Google Cloud Storage location where the input will be read from.

  Fields:
    uri: Google Cloud Storage URI for the input file. This must only be a
      Google Cloud Storage object. Wildcards are not currently supported.
  """
    uri = _messages.StringField(1)