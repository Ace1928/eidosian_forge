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
def _strip_and_collapse_whitespace(lexical_or_value: _AnyT) -> _AnyT:
    if isinstance(lexical_or_value, str):
        return re.sub(' +', ' ', lexical_or_value.strip())
    return lexical_or_value