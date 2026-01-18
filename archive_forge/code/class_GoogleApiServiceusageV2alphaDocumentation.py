from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV2alphaDocumentation(_messages.Message):
    """Provides more detailed information about a service, such as tutorials
  and pricing information.

  Fields:
    learnmoreUrl: Provides a URL where service consumers can learn more about
      the product.
    pricingUrl: Provides a link to pricing information for the service, such
      as https://cloud.google.com/bigquery/pricing.
  """
    learnmoreUrl = _messages.StringField(1)
    pricingUrl = _messages.StringField(2)