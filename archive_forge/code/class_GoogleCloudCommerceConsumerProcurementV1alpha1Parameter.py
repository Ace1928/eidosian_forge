from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Parameter(_messages.Message):
    """User-provided Parameters.

  Fields:
    name: Name of the parameter.
    value: Value of parameter.
  """
    name = _messages.StringField(1)
    value = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1ParameterValue', 2)