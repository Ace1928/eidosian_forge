from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesCustomerSettingsPatchRequest(_messages.Message):
    """A ServicemanagementServicesCustomerSettingsPatchRequest object.

  Fields:
    customerId: ID for the customer. See the comment for
      `CustomerSettings.customer_id` field of message for its format. This
      field is required.
    customerSettings: A CustomerSettings resource to be passed as the request
      body.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`. This
      field is required.
    updateMask: The field mask specifying which fields are to be updated.
  """
    customerId = _messages.StringField(1, required=True)
    customerSettings = _messages.MessageField('CustomerSettings', 2)
    serviceName = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)