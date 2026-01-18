from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeBackendBucketsGetRequest(_messages.Message):
    """A ComputeBackendBucketsGetRequest object.

  Fields:
    backendBucket: Name of the BackendBucket resource to return.
    project: Project ID for this request.
  """
    backendBucket = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)