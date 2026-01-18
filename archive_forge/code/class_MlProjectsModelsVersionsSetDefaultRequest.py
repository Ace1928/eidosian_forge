from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsVersionsSetDefaultRequest(_messages.Message):
    """A MlProjectsModelsVersionsSetDefaultRequest object.

  Fields:
    googleCloudMlV1SetDefaultVersionRequest: A
      GoogleCloudMlV1SetDefaultVersionRequest resource to be passed as the
      request body.
    name: Required. The name of the version to make the default for the model.
      You can get the names of all the versions of a model by calling
      projects.models.versions.list.
  """
    googleCloudMlV1SetDefaultVersionRequest = _messages.MessageField('GoogleCloudMlV1SetDefaultVersionRequest', 1)
    name = _messages.StringField(2, required=True)