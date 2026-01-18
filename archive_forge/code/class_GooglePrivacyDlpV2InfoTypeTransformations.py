from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeTransformations(_messages.Message):
    """A type of transformation that will scan unstructured text and apply
  various `PrimitiveTransformation`s to each finding, where the transformation
  is applied to only values that were identified as a specific info_type.

  Fields:
    transformations: Required. Transformation for each infoType. Cannot
      specify more than one for a given infoType.
  """
    transformations = _messages.MessageField('GooglePrivacyDlpV2InfoTypeTransformation', 1, repeated=True)