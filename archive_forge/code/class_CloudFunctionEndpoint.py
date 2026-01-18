from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudFunctionEndpoint(_messages.Message):
    """Wrapper for Cloud Function attributes.

  Fields:
    uri: A [Cloud Function](https://cloud.google.com/functions) name.
  """
    uri = _messages.StringField(1)