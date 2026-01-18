from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExclusionTypeValueValuesEnum(_messages.Enum):
    """If set to EXCLUSION_TYPE_EXCLUDE this infoType will not cause a
    finding to be returned. It still can be used for rules matching.

    Values:
      EXCLUSION_TYPE_UNSPECIFIED: A finding of this custom info type will not
        be excluded from results.
      EXCLUSION_TYPE_EXCLUDE: A finding of this custom info type will be
        excluded from final results, but can still affect rule execution.
    """
    EXCLUSION_TYPE_UNSPECIFIED = 0
    EXCLUSION_TYPE_EXCLUDE = 1