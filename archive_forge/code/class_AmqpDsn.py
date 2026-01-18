import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class AmqpDsn(AnyUrl):
    allowed_schemes = {'amqp', 'amqps'}
    host_required = False