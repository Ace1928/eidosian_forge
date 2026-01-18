from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesConsumerQuotaMetricsImportConsumerOverridesRequest(_messages.Message):
    """A ServiceusageServicesConsumerQuotaMetricsImportConsumerOverridesRequest
  object.

  Fields:
    importConsumerOverridesRequest: A ImportConsumerOverridesRequest resource
      to be passed as the request body.
    parent: The resource name of the consumer.  An example name would be:
      `projects/123/services/compute.googleapis.com`
  """
    importConsumerOverridesRequest = _messages.MessageField('ImportConsumerOverridesRequest', 1)
    parent = _messages.StringField(2, required=True)