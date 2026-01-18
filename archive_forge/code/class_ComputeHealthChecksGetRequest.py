from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeHealthChecksGetRequest(_messages.Message):
    """A ComputeHealthChecksGetRequest object.

  Fields:
    healthCheck: Name of the HealthCheck resource to return.
    project: Project ID for this request.
  """
    healthCheck = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)