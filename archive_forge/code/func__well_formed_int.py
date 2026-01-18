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
def _well_formed_int(lexical: Union[str, bytes], value: Any) -> bool:
    """
    The value space of xs:int is the set of common single size integers (32 bits),
    i.e., the integers between -2147483648 and 2147483647,
    its lexical space allows any number of insignificant leading zeros.
    """
    return len(lexical) > 0 and isinstance(value, int) and (-2147483648 <= value <= 2147483647)