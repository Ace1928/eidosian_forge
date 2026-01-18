from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsVersionsCreateRequest(_messages.Message):
    """A MlProjectsModelsVersionsCreateRequest object.

  Fields:
    googleCloudMlV1Version: A GoogleCloudMlV1Version resource to be passed as
      the request body.
    parent: Required. The name of the model.
  """
    googleCloudMlV1Version = _messages.MessageField('GoogleCloudMlV1Version', 1)
    parent = _messages.StringField(2, required=True)