from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1TestConfig(_messages.Message):
    """The test configuration for the resource.

  Fields:
    isTesting: Whether the resource is for testing or not.
  """
    isTesting = _messages.BooleanField(1)