from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceconsumermanagementServicesConsumerQuotaMetricsImportProducerOverridesRequest(_messages.Message):
    """A ServiceconsumermanagementServicesConsumerQuotaMetricsImportProducerOve
  rridesRequest object.

  Fields:
    parent: The resource name of the consumer.  An example name would be:
      `services/compute.googleapis.com/projects/123`
    v1Beta1ImportProducerOverridesRequest: A
      V1Beta1ImportProducerOverridesRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    v1Beta1ImportProducerOverridesRequest = _messages.MessageField('V1Beta1ImportProducerOverridesRequest', 2)