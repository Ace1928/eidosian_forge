from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1AuditLog(_messages.Message):
    """Consumer Procurement Order Audit Log To be deprecated

  Fields:
    auditLogRecords: List of audit log records for an offer
    name: The resource name of the order auditLog Format:
      `billingAccounts/{billing_account}/orders/{order}/auditLog`
  """
    auditLogRecords = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1AuditLogRecord', 1, repeated=True)
    name = _messages.StringField(2)