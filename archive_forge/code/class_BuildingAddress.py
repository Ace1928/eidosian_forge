from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildingAddress(_messages.Message):
    """JSON template for the postal address of a building in Directory API.

  Fields:
    addressLines: Unstructured address lines describing the lower levels of an
      address.
    administrativeArea: Optional. Highest administrative subdivision which is
      used for postal addresses of a country or region.
    languageCode: Optional. BCP-47 language code of the contents of this
      address (if known).
    locality: Optional. Generally refers to the city/town portion of the
      address. Examples: US city, IT comune, UK post town. In regions of the
      world where localities are not well defined or do not fit into this
      structure well, leave locality empty and use addressLines.
    postalCode: Optional. Postal code of the address.
    regionCode: Required. CLDR region code of the country/region of the
      address.
    sublocality: Optional. Sublocality of the address.
  """
    addressLines = _messages.StringField(1, repeated=True)
    administrativeArea = _messages.StringField(2)
    languageCode = _messages.StringField(3)
    locality = _messages.StringField(4)
    postalCode = _messages.StringField(5)
    regionCode = _messages.StringField(6)
    sublocality = _messages.StringField(7)