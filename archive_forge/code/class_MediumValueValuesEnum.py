from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediumValueValuesEnum(_messages.Enum):
    """The medium on which the data is stored. Acceptable values today is
    only MEMORY or none. When none, the default will currently be backed by
    memory but could change over time. +optional

    Values:
      MEDIUM_UNSPECIFIED: When not specified, falls back to the default
        implementation which is currently in memory (this may change over
        time).
      MEMORY: Explicitly set the EmptyDir to be in memory. Uses tmpfs.
    """
    MEDIUM_UNSPECIFIED = 0
    MEMORY = 1