from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRoutersGetRouterStatusRequest(_messages.Message):
    """A ComputeRoutersGetRouterStatusRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region for this request.
    router: Name of the Router resource to query.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    router = _messages.StringField(3, required=True)