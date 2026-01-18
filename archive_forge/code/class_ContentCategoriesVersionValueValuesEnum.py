from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContentCategoriesVersionValueValuesEnum(_messages.Enum):
    """The content categories used for classification.

    Values:
      CONTENT_CATEGORIES_VERSION_UNSPECIFIED: If `ContentCategoriesVersion` is
        not specified, this option will default to `V1`.
      V1: Legacy content categories of our initial launch in 2017.
      V2: Updated content categories in 2022.
    """
    CONTENT_CATEGORIES_VERSION_UNSPECIFIED = 0
    V1 = 1
    V2 = 2