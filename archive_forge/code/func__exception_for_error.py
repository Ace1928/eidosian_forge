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
def _exception_for_error(self, status: int, content_type: str, body: str, uri: str) -> GeoIP2Error:
    if 400 <= status < 500:
        return self._exception_for_4xx_status(status, content_type, body, uri)
    if 500 <= status < 600:
        return self._exception_for_5xx_status(status, uri, body)
    return self._exception_for_non_200_status(status, uri, body)