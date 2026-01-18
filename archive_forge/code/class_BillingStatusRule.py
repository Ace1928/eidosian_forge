from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingStatusRule(_messages.Message):
    """Defines the billing status requirements for operations.  When used with
  [Service Control API](https://cloud.google.com/service-control/), the
  following statuses are supported:  - **current**: the associated billing
  account is up to date and capable of                paying for resource
  usages. - **delinquent**: the associated billing account has a correctable
  problem,                   such as late payment.  Mostly services should
  only allow `current` status when serving requests. In addition, services can
  choose to allow both `current` and `delinquent` statuses when serving read-
  only requests to resources. If the list of allowed_statuses is empty, it
  means no billing requirement.

  Fields:
    allowedStatuses: Allowed billing statuses. The billing status check passes
      if the actual billing status matches any of the provided values here.
    selector: Selects the operation names to which this rule applies. Refer to
      selector for syntax details.
  """
    allowedStatuses = _messages.StringField(1, repeated=True)
    selector = _messages.StringField(2)