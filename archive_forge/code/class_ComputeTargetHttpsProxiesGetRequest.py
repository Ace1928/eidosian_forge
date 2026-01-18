from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetHttpsProxiesGetRequest(_messages.Message):
    """A ComputeTargetHttpsProxiesGetRequest object.

  Fields:
    project: Project ID for this request.
    targetHttpsProxy: Name of the TargetHttpsProxy resource to return.
  """
    project = _messages.StringField(1, required=True)
    targetHttpsProxy = _messages.StringField(2, required=True)