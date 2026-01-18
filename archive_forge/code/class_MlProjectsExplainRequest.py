from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsExplainRequest(_messages.Message):
    """A MlProjectsExplainRequest object.

  Fields:
    googleCloudMlV1ExplainRequest: A GoogleCloudMlV1ExplainRequest resource to
      be passed as the request body.
    name: Required. The resource name of a model or a version. Authorization:
      requires the `predict` permission on the specified resource.
  """
    googleCloudMlV1ExplainRequest = _messages.MessageField('GoogleCloudMlV1ExplainRequest', 1)
    name = _messages.StringField(2, required=True)