from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockTypeValueValuesEnum(_messages.Enum):
    """Detected block type (text, image etc) for this block.

    Values:
      UNKNOWN: Unknown block type.
      TEXT: Regular text block.
      TABLE: Table block.
      PICTURE: Image block.
      RULER: Horizontal/vertical line box.
      BARCODE: Barcode block.
    """
    UNKNOWN = 0
    TEXT = 1
    TABLE = 2
    PICTURE = 3
    RULER = 4
    BARCODE = 5