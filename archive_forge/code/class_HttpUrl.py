import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class HttpUrl(AnyHttpUrl):
    tld_required = True
    max_length = 2083
    hidden_parts = {'port'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {'port': '80' if parts['scheme'] == 'http' else '443'}