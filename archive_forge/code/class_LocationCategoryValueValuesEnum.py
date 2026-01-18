from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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