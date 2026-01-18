from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetPoolsGetRequest(_messages.Message):
    """A ComputeTargetPoolsGetRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region scoping this request.
    targetPool: Name of the TargetPool resource to return.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    targetPool = _messages.StringField(3, required=True)