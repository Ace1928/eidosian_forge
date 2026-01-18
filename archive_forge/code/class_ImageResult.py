from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageResult(_messages.Message):
    """Result of evaluating one image.

  Enums:
    VerdictValueValuesEnum: The result of evaluating this image.

  Fields:
    allowlistResult: If the image was exempted by a top-level allow_pattern,
      contains the allowlist pattern that the image name matched.
    checkSetResult: If a check set was evaluated, contains the result of the
      check set. Empty if there were no check sets.
    explanation: Explanation of this image result. Only populated if no check
      sets were evaluated.
    imageUri: Image URI from the request.
    verdict: The result of evaluating this image.
  """

    class VerdictValueValuesEnum(_messages.Enum):
        """The result of evaluating this image.

    Values:
      IMAGE_VERDICT_UNSPECIFIED: Not specified. This should never be used.
      CONFORMANT: Image conforms to the policy.
      NON_CONFORMANT: Image does not conform to the policy.
      ERROR: Error evaluating the image. Non-conformance has precedence over
        errors.
    """
        IMAGE_VERDICT_UNSPECIFIED = 0
        CONFORMANT = 1
        NON_CONFORMANT = 2
        ERROR = 3
    allowlistResult = _messages.MessageField('AllowlistResult', 1)
    checkSetResult = _messages.MessageField('CheckSetResult', 2)
    explanation = _messages.StringField(3)
    imageUri = _messages.StringField(4)
    verdict = _messages.EnumField('VerdictValueValuesEnum', 5)