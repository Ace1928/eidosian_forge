from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceRestrictionValueValuesEnum(_messages.Enum):
    """Whether to enforce traffic restrictions based on `sources` field. If
    the `sources` fields is non-empty, then this field must be set to
    `SOURCE_RESTRICTION_ENABLED`.

    Values:
      SOURCE_RESTRICTION_UNSPECIFIED: Enforcement preference unspecified, will
        not enforce traffic restrictions based on `sources` in EgressFrom.
      SOURCE_RESTRICTION_ENABLED: Enforcement preference enabled, traffic
        restrictions will be enforced based on `sources` in EgressFrom.
      SOURCE_RESTRICTION_DISABLED: Enforcement preference disabled, will not
        enforce traffic restrictions based on `sources` in EgressFrom.
    """
    SOURCE_RESTRICTION_UNSPECIFIED = 0
    SOURCE_RESTRICTION_ENABLED = 1
    SOURCE_RESTRICTION_DISABLED = 2