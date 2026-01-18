from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CropHintsParams(_messages.Message):
    """Parameters for crop hints annotation request.

  Fields:
    aspectRatios: Aspect ratios in floats, representing the ratio of the width
      to the height of the image. For example, if the desired aspect ratio is
      4/3, the corresponding float value should be 1.33333. If not specified,
      the best possible crop is returned. The number of provided aspect ratios
      is limited to a maximum of 16; any aspect ratios provided after the 16th
      are ignored.
  """
    aspectRatios = _messages.FloatField(1, repeated=True, variant=_messages.Variant.FLOAT)