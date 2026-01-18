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
def _writeXML(xmlnode: Union[xml.dom.minidom.Document, xml.dom.minidom.DocumentFragment]) -> bytes:
    if isinstance(xmlnode, xml.dom.minidom.DocumentFragment):
        d = xml.dom.minidom.Document()
        d.childNodes += xmlnode.childNodes
        xmlnode = d
    s = xmlnode.toxml('utf-8')
    if s.startswith('<?xml version="1.0" encoding="utf-8"?>'.encode('latin-1')):
        s = s[38:]
    if s.startswith('<rdflibtoplevelelement>'.encode('latin-1')):
        s = s[23:-24]
    if s == '<rdflibtoplevelelement/>'.encode('latin-1'):
        s = ''.encode('latin-1')
    return s