from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerCompositeTypesPatchRequest(_messages.Message):
    """A DeploymentmanagerCompositeTypesPatchRequest object.

  Fields:
    compositeType: The name of the composite type for this request.
    compositeTypeResource: A CompositeType resource to be passed as the
      request body.
    project: The project ID for this request.
  """
    compositeType = _messages.StringField(1, required=True)
    compositeTypeResource = _messages.MessageField('CompositeType', 2)
    project = _messages.StringField(3, required=True)