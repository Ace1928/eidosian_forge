from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerTypeProvidersPatchRequest(_messages.Message):
    """A DeploymentmanagerTypeProvidersPatchRequest object.

  Fields:
    project: The project ID for this request.
    typeProvider: The name of the type provider for this request.
    typeProviderResource: A TypeProvider resource to be passed as the request
      body.
  """
    project = _messages.StringField(1, required=True)
    typeProvider = _messages.StringField(2, required=True)
    typeProviderResource = _messages.MessageField('TypeProvider', 3)