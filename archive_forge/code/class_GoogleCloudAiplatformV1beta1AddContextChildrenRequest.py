from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AddContextChildrenRequest(_messages.Message):
    """Request message for MetadataService.AddContextChildren.

  Fields:
    childContexts: The resource names of the child Contexts.
  """
    childContexts = _messages.StringField(1, repeated=True)