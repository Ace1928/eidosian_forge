from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CleanupPolicyCondition(_messages.Message):
    """CleanupPolicyCondition is a set of conditions attached to a
  CleanupPolicy. If multiple entries are set, all must be satisfied for the
  condition to be satisfied.

  Enums:
    TagStateValueValuesEnum: Match versions by tag status.

  Fields:
    moreStaleThan: Match versions that have not been pulled in the duration.
    newerThan: Match versions newer than a duration.
    olderThan: Match versions older than a duration.
    packageNamePrefixes: Match versions by package prefix. Applied on any
      prefix match.
    tagPrefixes: Match versions by tag prefix. Applied on any prefix match.
    tagState: Match versions by tag status.
    versionNamePrefixes: Match versions by version name prefix. Applied on any
      prefix match.
  """

    class TagStateValueValuesEnum(_messages.Enum):
        """Match versions by tag status.

    Values:
      TAG_STATE_UNSPECIFIED: Tag status not specified.
      TAGGED: Applies to tagged versions only.
      UNTAGGED: Applies to untagged versions only.
      ANY: Applies to all versions.
    """
        TAG_STATE_UNSPECIFIED = 0
        TAGGED = 1
        UNTAGGED = 2
        ANY = 3
    moreStaleThan = _messages.StringField(1)
    newerThan = _messages.StringField(2)
    olderThan = _messages.StringField(3)
    packageNamePrefixes = _messages.StringField(4, repeated=True)
    tagPrefixes = _messages.StringField(5, repeated=True)
    tagState = _messages.EnumField('TagStateValueValuesEnum', 6)
    versionNamePrefixes = _messages.StringField(7, repeated=True)