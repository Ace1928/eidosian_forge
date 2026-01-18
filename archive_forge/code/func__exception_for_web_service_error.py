import ipaddress
import json
from typing import Any, Dict, cast, List, Optional, Type, Union
import aiohttp
import aiohttp.http
import requests
import requests.utils
import geoip2
import geoip2.models
from geoip2.errors import (
from geoip2.models import City, Country, Insights
from geoip2.types import IPAddress
@staticmethod
def _exception_for_web_service_error(message: str, code: str, status: int, uri: str) -> Union[AuthenticationError, AddressNotFoundError, PermissionRequiredError, OutOfQueriesError, InvalidRequestError]:
    if code in ('IP_ADDRESS_NOT_FOUND', 'IP_ADDRESS_RESERVED'):
        return AddressNotFoundError(message)
    if code in ('ACCOUNT_ID_REQUIRED', 'ACCOUNT_ID_UNKNOWN', 'AUTHORIZATION_INVALID', 'LICENSE_KEY_REQUIRED', 'USER_ID_REQUIRED', 'USER_ID_UNKNOWN'):
        return AuthenticationError(message)
    if code in ('INSUFFICIENT_FUNDS', 'OUT_OF_QUERIES'):
        return OutOfQueriesError(message)
    if code == 'PERMISSION_REQUIRED':
        return PermissionRequiredError(message)
    return InvalidRequestError(message, code, status, uri)