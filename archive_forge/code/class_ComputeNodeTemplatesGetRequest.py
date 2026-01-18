from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNodeTemplatesGetRequest(_messages.Message):
    """A ComputeNodeTemplatesGetRequest object.

  Fields:
    nodeTemplate: Name of the node template to return.
    project: Project ID for this request.
    region: The name of the region for this request.
  """
    nodeTemplate = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)