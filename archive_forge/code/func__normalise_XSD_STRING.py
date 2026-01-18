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
def _normalise_XSD_STRING(lexical_or_value: _AnyT) -> _AnyT:
    """
    Replaces 	, 
, \r (#x9 (tab), #xA (linefeed), and #xD (carriage return)) with space without any whitespace collapsing
    """
    if isinstance(lexical_or_value, str):
        return lexical_or_value.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    return lexical_or_value