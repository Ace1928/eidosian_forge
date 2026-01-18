from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeaturestoreOnlineServingConfig(_messages.Message):
    """OnlineServingConfig specifies the details for provisioning online
  serving resources.

  Fields:
    fixedNodeCount: The number of nodes for the online store. The number of
      nodes doesn't scale automatically, but you can manually update the
      number of nodes. If set to 0, the featurestore will not have an online
      store and cannot be used for online serving.
    scaling: Online serving scaling configuration. Only one of
      `fixed_node_count` and `scaling` can be set. Setting one will reset the
      other.
  """
    fixedNodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    scaling = _messages.MessageField('GoogleCloudAiplatformV1FeaturestoreOnlineServingConfigScaling', 2)