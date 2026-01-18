from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsConsentsGrantRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsConsentsGrantRequest object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1GrantConsentRequest: A
      GoogleCloudCommerceConsumerProcurementV1alpha1GrantConsentRequest
      resource to be passed as the request body.
    parent: Required. Parent of the consent to grant. Current supported format
      includes: - billingAccounts/{billing_account} - projects/{project_id}
  """
    googleCloudCommerceConsumerProcurementV1alpha1GrantConsentRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1GrantConsentRequest', 1)
    parent = _messages.StringField(2, required=True)