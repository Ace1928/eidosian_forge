import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class HostParts(TypedDict, total=False):
    host: str
    tld: Optional[str]
    host_type: Optional[str]
    port: Optional[str]
    rebuild: bool