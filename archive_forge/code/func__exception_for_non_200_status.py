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
def _exception_for_non_200_status(status: int, uri: str, body: Optional[str]) -> HTTPError:
    return HTTPError(f'Received a very surprising HTTP status ({status}) for {uri}', status, uri, body)