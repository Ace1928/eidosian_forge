from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ImageTransformations(_messages.Message):
    """A type of transformation that is applied over images.

  Fields:
    transforms: List of transforms to make.
  """
    transforms = _messages.MessageField('GooglePrivacyDlpV2ImageTransformation', 1, repeated=True)