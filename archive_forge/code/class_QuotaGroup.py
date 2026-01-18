from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaGroup(_messages.Message):
    """`QuotaGroup` defines a set of quota limits to enforce.

  Fields:
    billable: Indicates if the quota limits defined in this quota group apply
      to consumers who have active billing. Quota limits defined in billable
      groups will be applied only to consumers who have active billing. The
      amount of tokens consumed from billable quota group will also be
      reported for billing. Quota limits defined in non-billable groups will
      be applied only to consumers who have no active billing.
    description: User-visible description of this quota group.
    limits: Quota limits to be enforced when this quota group is used. A
      request must satisfy all the limits in a group for it to be permitted.
    name: Name of this quota group. Must be unique within the service.  Quota
      group name is used as part of the id for quota limits. Once the quota
      group has been put into use, the name of the quota group should be
      immutable.
  """
    billable = _messages.BooleanField(1)
    description = _messages.StringField(2)
    limits = _messages.MessageField('QuotaLimit', 3, repeated=True)
    name = _messages.StringField(4)