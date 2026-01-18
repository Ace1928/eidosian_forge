from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRoutesGetRequest(_messages.Message):
    """A ComputeRoutesGetRequest object.

  Fields:
    project: Project ID for this request.
    route: Name of the Route resource to return.
  """
    project = _messages.StringField(1, required=True)
    route = _messages.StringField(2, required=True)