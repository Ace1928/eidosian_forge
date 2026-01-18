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
def _well_formed_short(lexical: Union[str, bytes], value: Any) -> bool:
    """
    The value space of xs:short is the set of common short integers (16 bits),
    i.e., the integers between -32768 and 32767,
    its lexical space allows any number of insignificant leading zeros.
    """
    return len(lexical) > 0 and isinstance(value, int) and (-32768 <= value <= 32767)