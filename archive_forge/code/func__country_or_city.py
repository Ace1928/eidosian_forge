import socket
import geoip2.database
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_ipv46_address
from django.utils._os import to_path
from .resources import City, Country
@property
def _country_or_city(self):
    if self._country:
        return self._country.country
    else:
        return self._city.city