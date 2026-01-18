from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PurgeContextsMetadata(_messages.Message):
    """Details of operations that perform MetadataService.PurgeContexts.

  Fields:
    genericMetadata: Operation metadata for purging Contexts.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)