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
def _parse_html(lexical_form: str) -> xml.dom.minidom.DocumentFragment:
    """
    Parse the lexical form of an HTML literal into a document fragment
    using the ``dom`` from html5lib tree builder.

    :param lexical_form: The lexical form of the HTML literal.
    :return: A document fragment representing the HTML literal.
    :raises: `html5lib.html5parser.ParseError` if the lexical form is
        not valid HTML.
    """
    parser = html5lib.HTMLParser(tree=html5lib.treebuilders.getTreeBuilder('dom'), strict=True)
    result: xml.dom.minidom.DocumentFragment = parser.parseFragment(lexical_form)
    result.normalize()
    return result