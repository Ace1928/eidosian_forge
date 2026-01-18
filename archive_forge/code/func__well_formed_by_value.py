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
def _well_formed_by_value(lexical: Union[str, bytes], value: Any) -> bool:
    """
    This function is used as the fallback for detecting ill-typed/ill-formed
    literals and operates on the asumption that if a value (i.e.
    `Literal.value`) could be determined for a Literal then it is not
    ill-typed/ill-formed.

    This function will be called with `Literal.lexical` and `Literal.value` as arguments.
    """
    return value is not None