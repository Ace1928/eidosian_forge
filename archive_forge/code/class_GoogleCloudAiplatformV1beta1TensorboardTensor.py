from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TensorboardTensor(_messages.Message):
    """One point viewable on a tensor metric plot.

  Fields:
    value: Required. Serialized form of https://github.com/tensorflow/tensorfl
      ow/blob/master/tensorflow/core/framework/tensor.proto
    versionNumber: Optional. Version number of TensorProto used to serialize
      value.
  """
    value = _messages.BytesField(1)
    versionNumber = _messages.IntegerField(2, variant=_messages.Variant.INT32)