from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1Rule(_messages.Message):
    """A rule to be applied in a Policy.

  Enums:
    ActionValueValuesEnum: Required

  Fields:
    action: Required
    conditions: Additional restrictions that must be met. All conditions must
      pass for the rule to match.
    description: Human-readable description of the rule.
    in_: If one or more 'in' clauses are specified, the rule matches if the
      PRINCIPAL/AUTHORITY_SELECTOR is in at least one of these entries.
    logConfig: The config returned to callers of CheckPolicy for any entries
      that match the LOG action.
    notIn: If one or more 'not_in' clauses are specified, the rule matches if
      the PRINCIPAL/AUTHORITY_SELECTOR is in none of the entries. The format
      for in and not_in entries can be found at in the Local IAM documentation
      (see go/local-iam#features).
    permissions: A permission is a string of form '..' (e.g.,
      'storage.buckets.list'). A value of '*' matches all permissions, and a
      verb part of '*' (e.g., 'storage.buckets.*') matches all verbs.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required

    Values:
      NO_ACTION: Default no action.
      ALLOW: Matching 'Entries' grant access.
      ALLOW_WITH_LOG: Matching 'Entries' grant access and the caller promises
        to log the request per the returned log_configs.
      DENY: Matching 'Entries' deny access.
      DENY_WITH_LOG: Matching 'Entries' deny access and the caller promises to
        log the request per the returned log_configs.
      LOG: Matching 'Entries' tell IAM.Check callers to generate logs.
    """
        NO_ACTION = 0
        ALLOW = 1
        ALLOW_WITH_LOG = 2
        DENY = 3
        DENY_WITH_LOG = 4
        LOG = 5
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    conditions = _messages.MessageField('GoogleIamV1Condition', 2, repeated=True)
    description = _messages.StringField(3)
    in_ = _messages.StringField(4, repeated=True)
    logConfig = _messages.MessageField('GoogleIamV1LogConfig', 5, repeated=True)
    notIn = _messages.StringField(6, repeated=True)
    permissions = _messages.StringField(7, repeated=True)