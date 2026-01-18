import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class Subdivision(PlaceRecord):
    """Contains data for the subdivisions associated with an IP address.

    This class contains the subdivision data associated with an IP address.

    This attribute is returned by ``city``, ``enterprise``, and ``insights``.

    Attributes:

    .. attribute:: confidence

      This is a value from 0-100 indicating MaxMind's
      confidence that the subdivision is correct. This attribute is only
      available from the Insights end point and the Enterprise database.

      :type: int

    .. attribute:: geoname_id

      This is a GeoName ID for the subdivision.

      :type: int

    .. attribute:: iso_code

      This is a string up to three characters long
      contain the subdivision portion of the `ISO 3166-2 code
      <https://en.wikipedia.org/wiki/ISO_3166-2>`_.

      :type: str

    .. attribute:: name

      The name of the subdivision based on the locales list passed to the
      constructor.

      :type: str

    .. attribute:: names

      A dictionary where the keys are locale codes and the
      values are names

      :type: dict

    """
    confidence: Optional[int]
    geoname_id: Optional[int]
    iso_code: Optional[str]

    def __init__(self, locales: Optional[List[str]]=None, confidence: Optional[int]=None, geoname_id: Optional[int]=None, iso_code: Optional[str]=None, names: Optional[Dict[str, str]]=None, **_) -> None:
        self.confidence = confidence
        self.geoname_id = geoname_id
        self.iso_code = iso_code
        super().__init__(locales, names)