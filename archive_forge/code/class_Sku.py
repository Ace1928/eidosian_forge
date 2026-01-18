from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Sku(_messages.Message):
    """Encapsulates a single SKU in Google Cloud Platform

  Fields:
    category: The category hierarchy of this SKU, purely for organizational
      purpose.
    description: A human readable description of the SKU, has a maximum length
      of 256 characters.
    geoTaxonomy: The geographic taxonomy for this sku.
    name: The resource name for the SKU. Example:
      "services/DA34-426B-A397/skus/AA95-CD31-42FE"
    pricingInfo: A timeline of pricing info for this SKU in chronological
      order.
    serviceProviderName: Identifies the service provider. This is 'Google' for
      first party services in Google Cloud Platform.
    serviceRegions: List of service regions this SKU is offered at. Example:
      "asia-east1" Service regions can be found at
      https://cloud.google.com/about/locations/
    skuId: The identifier for the SKU. Example: "AA95-CD31-42FE"
  """
    category = _messages.MessageField('Category', 1)
    description = _messages.StringField(2)
    geoTaxonomy = _messages.MessageField('GeoTaxonomy', 3)
    name = _messages.StringField(4)
    pricingInfo = _messages.MessageField('PricingInfo', 5, repeated=True)
    serviceProviderName = _messages.StringField(6)
    serviceRegions = _messages.StringField(7, repeated=True)
    skuId = _messages.StringField(8)