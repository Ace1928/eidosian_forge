from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SummaryResult(_messages.Message):
    """A collection that informs the user the number of times a particular
  `TransformationResultCode` and error details occurred.

  Enums:
    CodeValueValuesEnum: Outcome of the transformation.

  Fields:
    code: Outcome of the transformation.
    count: Number of transformations counted by this result.
    details: A place for warnings or errors to show up if a transformation
      didn't work as expected.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Outcome of the transformation.

    Values:
      TRANSFORMATION_RESULT_CODE_UNSPECIFIED: Unused
      SUCCESS: Transformation completed without an error.
      ERROR: Transformation had an error.
    """
        TRANSFORMATION_RESULT_CODE_UNSPECIFIED = 0
        SUCCESS = 1
        ERROR = 2
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    count = _messages.IntegerField(2)
    details = _messages.StringField(3)