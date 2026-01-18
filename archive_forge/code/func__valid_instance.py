import base64
import calendar
from ipaddress import AddressValueError
from ipaddress import IPv4Address
from ipaddress import IPv6Address
import re
import struct
import time
from urllib.parse import urlparse
from saml2 import time_util
def _valid_instance(instance, val):
    try:
        val.verify()
    except NotValid as exc:
        raise NotValid(f"Class '{instance.__class__.__name__}' instance: {exc.args[0]}")
    except OutsideCardinality as exc:
        raise NotValid(f"Class '{instance.__class__.__name__}' instance cardinality error: {exc.args[0]}")