from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomerSettings(_messages.Message):
    """Settings that control how a customer (identified by a billing account)
  uses a service

  Fields:
    customerId: ID for the customer that consumes the service (see above). The
      supported types of customers are:  1. domain:{domain} A Google Apps
      domain name. For example, google.com.  2.
      billingAccount:{billing_account_id} A Google Cloud Plafrom billing
      account. For Example, 123456-7890ab-cdef12.
    quotaSettings: Settings that control how much or how fast the service can
      be used by the consumer projects owned by the customer collectively.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.
  """
    customerId = _messages.StringField(1)
    quotaSettings = _messages.MessageField('QuotaSettings', 2)
    serviceName = _messages.StringField(3)