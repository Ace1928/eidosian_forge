from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VariableTermQuota(_messages.Message):
    """A variable term quota is a bucket of tokens that is consumed over a
  specified (usually long) time period. When present, it overrides any "1d"
  duration per-project quota specified on the group.  Variable terms run from
  midnight to midnight, start_date to end_date (inclusive) in the
  America/Los_Angeles time zone.

  Fields:
    createTime: Time when this variable term quota was created. If multiple
      quotas are simultaneously active, then the quota with the latest
      create_time is the effective one.
    displayEndDate: The displayed end of the active period for the variable
      term quota. This may be before the effective end to give the user a
      grace period. YYYYMMdd date format, e.g. 20140730.
    endDate: The effective end of the active period for the variable term
      quota (inclusive). This must be no more than 5 years after start_date.
      YYYYMMdd date format, e.g. 20140730.
    groupName: The quota group that has the variable term quota applied to it.
      This must be a google.api.QuotaGroup.name specified in the service
      configuration.
    limit: The number of tokens available during the configured term.
    quotaUsage: The usage data of this quota.
    startDate: The beginning of the active period for the variable term quota.
      YYYYMMdd date format, e.g. 20140730.
  """
    createTime = _messages.StringField(1)
    displayEndDate = _messages.StringField(2)
    endDate = _messages.StringField(3)
    groupName = _messages.StringField(4)
    limit = _messages.IntegerField(5)
    quotaUsage = _messages.MessageField('QuotaUsage', 6)
    startDate = _messages.StringField(7)