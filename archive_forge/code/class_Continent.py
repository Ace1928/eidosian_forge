import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class Continent(PlaceRecord):
    """Contains data for the continent record associated with an IP address.

    This class contains the continent-level data associated with an IP
    address.

    Attributes:


    .. attribute:: code

      A two character continent code like "NA" (North America)
      or "OC" (Oceania).

      :type: str

    .. attribute:: geoname_id

      The GeoName ID for the continent.

      :type: int

    .. attribute:: name

      Returns the name of the continent based on the locales list passed to
      the constructor.

      :type: str

    .. attribute:: names

      A dictionary where the keys are locale codes
      and the values are names.

      :type: dict

    """
    code: Optional[str]
    geoname_id: Optional[int]

    def __init__(self, locales: Optional[List[str]]=None, code: Optional[str]=None, geoname_id: Optional[int]=None, names: Optional[Dict[str, str]]=None, **_) -> None:
        self.code = code
        self.geoname_id = geoname_id
        super().__init__(locales, names)