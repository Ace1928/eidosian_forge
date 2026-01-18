from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteInput(_messages.Message):
    """Input parameters for preview of delete operation.

  Fields:
    deployment: Required. Name of existing deployment to preview its deletion.
      Format:
      `projects/{project}/locations/{location}/deployments/{deployment}`
  """
    deployment = _messages.StringField(1)