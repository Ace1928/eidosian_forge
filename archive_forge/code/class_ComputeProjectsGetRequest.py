from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeProjectsGetRequest(_messages.Message):
    """A ComputeProjectsGetRequest object.

  Fields:
    project: Project ID for this request.
  """
    project = _messages.StringField(1, required=True)