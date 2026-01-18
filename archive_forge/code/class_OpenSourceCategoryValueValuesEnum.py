from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OpenSourceCategoryValueValuesEnum(_messages.Enum):
    """Required. Indicates the open source category of the publisher model.

    Values:
      OPEN_SOURCE_CATEGORY_UNSPECIFIED: The open source category is
        unspecified, which should not be used.
      PROPRIETARY: Used to indicate the PublisherModel is not open sourced.
      GOOGLE_OWNED_OSS_WITH_GOOGLE_CHECKPOINT: Used to indicate the
        PublisherModel is a Google-owned open source model w/ Google
        checkpoint.
      THIRD_PARTY_OWNED_OSS_WITH_GOOGLE_CHECKPOINT: Used to indicate the
        PublisherModel is a 3p-owned open source model w/ Google checkpoint.
      GOOGLE_OWNED_OSS: Used to indicate the PublisherModel is a Google-owned
        pure open source model.
      THIRD_PARTY_OWNED_OSS: Used to indicate the PublisherModel is a 3p-owned
        pure open source model.
    """
    OPEN_SOURCE_CATEGORY_UNSPECIFIED = 0
    PROPRIETARY = 1
    GOOGLE_OWNED_OSS_WITH_GOOGLE_CHECKPOINT = 2
    THIRD_PARTY_OWNED_OSS_WITH_GOOGLE_CHECKPOINT = 3
    GOOGLE_OWNED_OSS = 4
    THIRD_PARTY_OWNED_OSS = 5