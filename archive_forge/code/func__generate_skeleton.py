import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def _generate_skeleton(self, shape, stack, name=''):
    stack.append(shape.name)
    try:
        if shape.type_name == 'structure':
            return self._generate_type_structure(shape, stack)
        elif shape.type_name == 'list':
            return self._generate_type_list(shape, stack)
        elif shape.type_name == 'map':
            return self._generate_type_map(shape, stack)
        elif shape.type_name == 'string':
            if self._use_member_names:
                return name
            if shape.enum:
                return random.choice(shape.enum)
            return ''
        elif shape.type_name in ['integer', 'long']:
            return 0
        elif shape.type_name in ['float', 'double']:
            return 0.0
        elif shape.type_name == 'boolean':
            return True
        elif shape.type_name == 'timestamp':
            return datetime.datetime(1970, 1, 1, 0, 0, 0)
    finally:
        stack.pop()