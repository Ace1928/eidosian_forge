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
def _parseBoolean(value: Union[str, bytes]) -> bool:
    """
    Boolean is a datatype with value space {true,false},
    lexical space {"true", "false","1","0"} and
    lexical-to-value mapping {"true"→true, "false"→false, "1"→true, "0"→false}.
    """
    true_accepted_values = ['1', 'true', b'1', b'true']
    false_accepted_values = ['0', 'false', b'0', b'false']
    new_value = value.lower()
    if new_value in true_accepted_values:
        return True
    if new_value not in false_accepted_values:
        warnings.warn('Parsing weird boolean, % r does not map to True or False' % value, category=UserWarning)
    return False