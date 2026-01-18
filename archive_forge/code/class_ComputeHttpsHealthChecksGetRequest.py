from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeHttpsHealthChecksGetRequest(_messages.Message):
    """A ComputeHttpsHealthChecksGetRequest object.

  Fields:
    httpsHealthCheck: Name of the HttpsHealthCheck resource to return.
    project: Project ID for this request.
  """
    httpsHealthCheck = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)