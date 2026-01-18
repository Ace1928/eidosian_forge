import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class CockroachDsn(AnyUrl):
    allowed_schemes = {'cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'}
    user_required = True