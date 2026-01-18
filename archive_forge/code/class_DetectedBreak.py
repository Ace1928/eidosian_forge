from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetectedBreak(_messages.Message):
    """Detected start or end of a structural component.

  Enums:
    TypeValueValuesEnum: Detected break type.

  Fields:
    isPrefix: True if break prepends the element.
    type: Detected break type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Detected break type.

    Values:
      UNKNOWN: Unknown break label type.
      SPACE: Regular space.
      SURE_SPACE: Sure space (very wide).
      EOL_SURE_SPACE: Line-wrapping break.
      HYPHEN: End-line hyphen that is not present in text; does not co-occur
        with `SPACE`, `LEADER_SPACE`, or `LINE_BREAK`.
      LINE_BREAK: Line break that ends a paragraph.
    """
        UNKNOWN = 0
        SPACE = 1
        SURE_SPACE = 2
        EOL_SURE_SPACE = 3
        HYPHEN = 4
        LINE_BREAK = 5
    isPrefix = _messages.BooleanField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)