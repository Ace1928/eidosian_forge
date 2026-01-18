import ipaddress
from abc import ABCMeta
from typing import Any, cast, Dict, List, Optional, Union
import geoip2.records
from geoip2.mixins import SimpleEquality
class ASN(SimpleModel):
    """Model class for the GeoLite2 ASN.

    This class provides the following attribute:

    .. attribute:: autonomous_system_number

      The autonomous system number associated with the IP address.

      :type: int

    .. attribute:: autonomous_system_organization

      The organization associated with the registered autonomous system number
      for the IP address.

      :type: str

    .. attribute:: ip_address

      The IP address used in the lookup.

      :type: str

    .. attribute:: network

      The network associated with the record. In particular, this is the
      largest network where all of the fields besides ip_address have the same
      value.

      :type: ipaddress.IPv4Network or ipaddress.IPv6Network
    """
    autonomous_system_number: Optional[int]
    autonomous_system_organization: Optional[str]

    def __init__(self, raw: Dict[str, Union[str, int]]) -> None:
        super().__init__(raw)
        self.autonomous_system_number = cast(Optional[int], raw.get('autonomous_system_number'))
        self.autonomous_system_organization = cast(Optional[str], raw.get('autonomous_system_organization'))