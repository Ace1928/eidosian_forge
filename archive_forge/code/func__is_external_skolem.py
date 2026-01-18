import re
from fractions import Fraction
import logging
import math
import warnings
import xml.dom.minidom
from base64 import b64decode, b64encode
from binascii import hexlify, unhexlify
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from re import compile, sub
from typing import (
from urllib.parse import urldefrag, urljoin, urlparse
from isodate import (
import rdflib
import rdflib.util
from rdflib.compat import long_type
@staticmethod
def _is_external_skolem(uri: Any) -> bool:
    if not isinstance(uri, str):
        uri = str(uri)
    parsed_uri = urlparse(uri)
    gen_id = parsed_uri.path.rfind(skolem_genid)
    if gen_id != 0:
        return False
    return True