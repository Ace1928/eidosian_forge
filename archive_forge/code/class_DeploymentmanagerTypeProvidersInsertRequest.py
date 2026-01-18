from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerTypeProvidersInsertRequest(_messages.Message):
    """A DeploymentmanagerTypeProvidersInsertRequest object.

  Fields:
    project: The project ID for this request.
    typeProvider: A TypeProvider resource to be passed as the request body.
  """
    project = _messages.StringField(1, required=True)
    typeProvider = _messages.MessageField('TypeProvider', 2)