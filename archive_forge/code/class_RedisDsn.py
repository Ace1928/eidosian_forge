import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class RedisDsn(AnyUrl):
    __slots__ = ()
    allowed_schemes = {'redis', 'rediss'}
    host_required = False

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {'domain': 'localhost' if not (parts['ipv4'] or parts['ipv6']) else '', 'port': '6379', 'path': '/0'}