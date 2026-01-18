from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsCreateRequest(_messages.Message):
    """A MlProjectsModelsCreateRequest object.

  Fields:
    googleCloudMlV1Model: A GoogleCloudMlV1Model resource to be passed as the
      request body.
    parent: Required. The project name.
  """
    googleCloudMlV1Model = _messages.MessageField('GoogleCloudMlV1Model', 1)
    parent = _messages.StringField(2, required=True)