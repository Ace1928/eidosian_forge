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
def _comparable_to(self, other: Any) -> bool:
    """
        Helper method to decide which things are meaningful to
        rich-compare with this literal
        """
    if isinstance(other, Literal):
        if self.datatype is not None and other.datatype is not None:
            if self.datatype not in XSDToPython or other.datatype not in XSDToPython:
                if self.datatype != other.datatype:
                    return False
        else:
            if not (self.datatype == _XSD_STRING and (not other.datatype)) or (other.datatype == _XSD_STRING and (not self.datatype)):
                return False
            if (self.language or '').lower() != (other.language or '').lower():
                return False
    return True