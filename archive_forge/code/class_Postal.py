import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class Postal(Record):
    """Contains data for the postal record associated with an IP address.

    This class contains the postal data associated with an IP address.

    This attribute is returned by ``city``, ``enterprise``, and ``insights``.

    Attributes:

    .. attribute:: code

      The postal code of the location. Postal
      codes are not available for all countries. In some countries, this will
      only contain part of the postal code.

      :type: str

    .. attribute:: confidence

      A value from 0-100 indicating
      MaxMind's confidence that the postal code is correct. This attribute is
      only available from the Insights end point and the Enterprise database.

      :type: int

    """
    code: Optional[str]
    confidence: Optional[int]

    def __init__(self, code: Optional[str]=None, confidence: Optional[int]=None, **_) -> None:
        self.code = code
        self.confidence = confidence