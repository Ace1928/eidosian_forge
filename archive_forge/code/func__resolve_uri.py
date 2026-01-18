from __future__ import annotations
import ipaddress
import random
from typing import Any, Optional, Union
from pymongo.common import CONNECT_TIMEOUT
from pymongo.errors import ConfigurationError
def _resolve_uri(self, encapsulate_errors: bool) -> resolver.Answer:
    try:
        results = _resolve('_' + self.__srv + '._tcp.' + self.__fqdn, 'SRV', lifetime=self.__connect_timeout)
    except Exception as exc:
        if not encapsulate_errors:
            raise
        raise ConfigurationError(str(exc)) from None
    return results