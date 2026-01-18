from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessLocations(_messages.Message):
    """Home office and physical location of the principal.

  Fields:
    principalOfficeCountry: The "home office" location of the principal. A
      two-letter country code (ISO 3166-1 alpha-2), such as "US", "DE" or "GB"
      or a region code. In some limited situations Google systems may refer
      refer to a region code instead of a country code. Possible Region Codes:
      * ASI: Asia * EUR: Europe * OCE: Oceania * AFR: Africa * NAM: North
      America * SAM: South America * ANT: Antarctica * ANY: Any location
    principalPhysicalLocationCountry: Physical location of the principal at
      the time of the access. A two-letter country code (ISO 3166-1 alpha-2),
      such as "US", "DE" or "GB" or a region code. In some limited situations
      Google systems may refer refer to a region code instead of a country
      code. Possible Region Codes: * ASI: Asia * EUR: Europe * OCE: Oceania *
      AFR: Africa * NAM: North America * SAM: South America * ANT: Antarctica
      * ANY: Any location
  """
    principalOfficeCountry = _messages.StringField(1)
    principalPhysicalLocationCountry = _messages.StringField(2)