from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSslPoliciesGetRequest(_messages.Message):
    """A ComputeSslPoliciesGetRequest object.

  Fields:
    project: Project ID for this request.
    sslPolicy: Name of the SSL policy to update. The name must be 1-63
      characters long, and comply with RFC1035.
  """
    project = _messages.StringField(1, required=True)
    sslPolicy = _messages.StringField(2, required=True)