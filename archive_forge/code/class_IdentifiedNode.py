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
class IdentifiedNode(Identifier):
    """
    An abstract class, primarily defined to identify Nodes that are not Literals.

    The name "Identified Node" is not explicitly defined in the RDF specification, but can be drawn from this section: https://www.w3.org/TR/rdf-concepts/#section-URI-Vocabulary
    """

    def __getnewargs__(self) -> Tuple[str]:
        return (str(self),)

    def toPython(self) -> str:
        return str(self)