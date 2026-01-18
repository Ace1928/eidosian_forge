from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeCategory(_messages.Message):
    """Classification of infoTypes to organize them according to geographic
  location, industry, and data type.

  Enums:
    IndustryCategoryValueValuesEnum: The group of relevant businesses where
      this infoType is commonly used
    LocationCategoryValueValuesEnum: The region or country that issued the ID
      or document represented by the infoType.
    TypeCategoryValueValuesEnum: The class of identifiers where this infoType
      belongs

  Fields:
    industryCategory: The group of relevant businesses where this infoType is
      commonly used
    locationCategory: The region or country that issued the ID or document
      represented by the infoType.
    typeCategory: The class of identifiers where this infoType belongs
  """

    class IndustryCategoryValueValuesEnum(_messages.Enum):
        """The group of relevant businesses where this infoType is commonly used

    Values:
      INDUSTRY_UNSPECIFIED: Unused industry
      FINANCE: The infoType is typically used in the finance industry.
      HEALTH: The infoType is typically used in the health industry.
      TELECOMMUNICATIONS: The infoType is typically used in the
        telecommunications industry.
    """
        INDUSTRY_UNSPECIFIED = 0
        FINANCE = 1
        HEALTH = 2
        TELECOMMUNICATIONS = 3

    class LocationCategoryValueValuesEnum(_messages.Enum):
        """The region or country that issued the ID or document represented by
    the infoType.

    Values:
      LOCATION_UNSPECIFIED: Unused location
      GLOBAL: The infoType is not issued by or tied to a specific region, but
        is used almost everywhere.
      ARGENTINA: The infoType is typically used in Argentina.
      AUSTRALIA: The infoType is typically used in Australia.
      BELGIUM: The infoType is typically used in Belgium.
      BRAZIL: The infoType is typically used in Brazil.
      CANADA: The infoType is typically used in Canada.
      CHILE: The infoType is typically used in Chile.
      CHINA: The infoType is typically used in China.
      COLOMBIA: The infoType is typically used in Colombia.
      CROATIA: The infoType is typically used in Croatia.
      DENMARK: The infoType is typically used in Denmark.
      FRANCE: The infoType is typically used in France.
      FINLAND: The infoType is typically used in Finland.
      GERMANY: The infoType is typically used in Germany.
      HONG_KONG: The infoType is typically used in Hong Kong.
      INDIA: The infoType is typically used in India.
      INDONESIA: The infoType is typically used in Indonesia.
      IRELAND: The infoType is typically used in Ireland.
      ISRAEL: The infoType is typically used in Israel.
      ITALY: The infoType is typically used in Italy.
      JAPAN: The infoType is typically used in Japan.
      KOREA: The infoType is typically used in Korea.
      MEXICO: The infoType is typically used in Mexico.
      THE_NETHERLANDS: The infoType is typically used in the Netherlands.
      NEW_ZEALAND: The infoType is typically used in New Zealand.
      NORWAY: The infoType is typically used in Norway.
      PARAGUAY: The infoType is typically used in Paraguay.
      PERU: The infoType is typically used in Peru.
      POLAND: The infoType is typically used in Poland.
      PORTUGAL: The infoType is typically used in Portugal.
      SINGAPORE: The infoType is typically used in Singapore.
      SOUTH_AFRICA: The infoType is typically used in South Africa.
      SPAIN: The infoType is typically used in Spain.
      SWEDEN: The infoType is typically used in Sweden.
      SWITZERLAND: The infoType is typically used in Switzerland.
      TAIWAN: The infoType is typically used in Taiwan.
      THAILAND: The infoType is typically used in Thailand.
      TURKEY: The infoType is typically used in Turkey.
      UNITED_KINGDOM: The infoType is typically used in the United Kingdom.
      UNITED_STATES: The infoType is typically used in the United States.
      URUGUAY: The infoType is typically used in Uruguay.
      VENEZUELA: The infoType is typically used in Venezuela.
      INTERNAL: The infoType is typically used in Google internally.
    """
        LOCATION_UNSPECIFIED = 0
        GLOBAL = 1
        ARGENTINA = 2
        AUSTRALIA = 3
        BELGIUM = 4
        BRAZIL = 5
        CANADA = 6
        CHILE = 7
        CHINA = 8
        COLOMBIA = 9
        CROATIA = 10
        DENMARK = 11
        FRANCE = 12
        FINLAND = 13
        GERMANY = 14
        HONG_KONG = 15
        INDIA = 16
        INDONESIA = 17
        IRELAND = 18
        ISRAEL = 19
        ITALY = 20
        JAPAN = 21
        KOREA = 22
        MEXICO = 23
        THE_NETHERLANDS = 24
        NEW_ZEALAND = 25
        NORWAY = 26
        PARAGUAY = 27
        PERU = 28
        POLAND = 29
        PORTUGAL = 30
        SINGAPORE = 31
        SOUTH_AFRICA = 32
        SPAIN = 33
        SWEDEN = 34
        SWITZERLAND = 35
        TAIWAN = 36
        THAILAND = 37
        TURKEY = 38
        UNITED_KINGDOM = 39
        UNITED_STATES = 40
        URUGUAY = 41
        VENEZUELA = 42
        INTERNAL = 43

    class TypeCategoryValueValuesEnum(_messages.Enum):
        """The class of identifiers where this infoType belongs

    Values:
      TYPE_UNSPECIFIED: Unused type
      PII: Personally identifiable information, for example, a name or phone
        number
      SPII: Personally identifiable information that is especially sensitive,
        for example, a passport number.
      DEMOGRAPHIC: Attributes that can partially identify someone, especially
        in combination with other attributes, like age, height, and gender.
      CREDENTIAL: Confidential or secret information, for example, a password.
      GOVERNMENT_ID: An identification document issued by a government.
      DOCUMENT: A document, for example, a resume or source code.
      CONTEXTUAL_INFORMATION: Information that is not sensitive on its own,
        but provides details about the circumstances surrounding an entity or
        an event.
    """
        TYPE_UNSPECIFIED = 0
        PII = 1
        SPII = 2
        DEMOGRAPHIC = 3
        CREDENTIAL = 4
        GOVERNMENT_ID = 5
        DOCUMENT = 6
        CONTEXTUAL_INFORMATION = 7
    industryCategory = _messages.EnumField('IndustryCategoryValueValuesEnum', 1)
    locationCategory = _messages.EnumField('LocationCategoryValueValuesEnum', 2)
    typeCategory = _messages.EnumField('TypeCategoryValueValuesEnum', 3)