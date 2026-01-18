import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class RuleSetEndpoint(NamedTuple):
    """A resolved endpoint object returned by a rule."""
    url: str
    properties: dict
    headers: dict