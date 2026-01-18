import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class RepresentedCountry(Country):
    """Contains data for the represented country associated with an IP address.

    This class contains the country-level data associated with an IP address
    for the IP's represented country. The represented country is the country
    represented by something like a military base.

    Attributes:


    .. attribute:: confidence

      A value from 0-100 indicating MaxMind's confidence that
      the country is correct. This attribute is only available from the
      Insights end point and the Enterprise database.

      :type: int

    .. attribute:: geoname_id

      The GeoName ID for the country.

      :type: int

    .. attribute:: is_in_european_union

      This is true if the country is a member state of the European Union.

      :type: bool

    .. attribute:: iso_code

      The two-character `ISO 3166-1
      <https://en.wikipedia.org/wiki/ISO_3166-1>`_ alpha code for the country.

      :type: str

    .. attribute:: name

      The name of the country based on the locales list passed to the
      constructor.

      :type: str

    .. attribute:: names

      A dictionary where the keys are locale codes and the values
      are names.

      :type: dict


    .. attribute:: type

      A string indicating the type of entity that is representing the
      country. Currently we only return ``military`` but this could expand to
      include other types in the future.

      :type: str

    """
    type: Optional[str]

    def __init__(self, locales: Optional[List[str]]=None, confidence: Optional[int]=None, geoname_id: Optional[int]=None, is_in_european_union: bool=False, iso_code: Optional[str]=None, names: Optional[Dict[str, str]]=None, type: Optional[str]=None, **_) -> None:
        self.type = type
        super().__init__(locales, confidence, geoname_id, is_in_european_union, iso_code, names)