from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerOperationsGetRequest(_messages.Message):
    """A DeploymentmanagerOperationsGetRequest object.

  Fields:
    operation: The name of the operation for this request.
    project: The project ID for this request.
  """
    operation = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)