from __future__ import annotations
import datetime as py_datetime  # naming conflict with function within this module
import hashlib
import math
import operator as pyop  # python operators
import random
import re
import uuid
import warnings
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from functools import reduce
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union, overload
from urllib.parse import quote
import isodate
from pyparsing import ParseResults
from rdflib.namespace import RDF, XSD
from rdflib.plugins.sparql.datatypes import (
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import (
def Builtin_TIMEZONE(e: Expr, ctx) -> Literal:
    """
    http://www.w3.org/TR/sparql11-query/#func-timezone

    :returns: the timezone part of arg as an xsd:dayTimeDuration.
    :raises: an error if there is no timezone.
    """
    dt = datetime(e.arg)
    if not dt.tzinfo:
        raise SPARQLError('datatime has no timezone: %r' % dt)
    delta = dt.utcoffset()
    d = delta.days
    s = delta.seconds
    neg = ''
    if d < 0:
        s = -24 * 60 * 60 * d - s
        d = 0
        neg = '-'
    h = s / (60 * 60)
    m = (s - h * 60 * 60) / 60
    s = s - h * 60 * 60 - m * 60
    tzdelta = '%sP%sT%s%s%s' % (neg, '%dD' % d if d else '', '%dH' % h if h else '', '%dM' % m if m else '', '%dS' % s if not d and (not h) and (not m) else '')
    return Literal(tzdelta, datatype=XSD.dayTimeDuration)