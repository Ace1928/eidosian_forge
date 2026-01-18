import inspect
import os
from typing import Any, AnyStr, cast, IO, List, Optional, Type, Union
import maxminddb
from maxminddb import (
import geoip2
import geoip2.models
import geoip2.errors
from geoip2.types import IPAddress
from geoip2.models import (
def _flat_model_for(self, model_class: Union[Type[Domain], Type[ISP], Type[ConnectionType], Type[ASN], Type[AnonymousIP]], types: str, ip_address: IPAddress) -> Union[ConnectionType, ISP, AnonymousIP, Domain, ASN]:
    record, prefix_len = self._get(types, ip_address)
    record['ip_address'] = ip_address
    record['prefix_len'] = prefix_len
    return model_class(record)