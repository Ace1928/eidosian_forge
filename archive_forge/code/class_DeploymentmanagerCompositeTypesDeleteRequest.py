from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerCompositeTypesDeleteRequest(_messages.Message):
    """A DeploymentmanagerCompositeTypesDeleteRequest object.

  Fields:
    compositeType: The name of the type for this request.
    project: The project ID for this request.
  """
    compositeType = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)