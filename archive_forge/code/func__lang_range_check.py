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
def _lang_range_check(range: Literal, lang: Literal) -> bool:
    """
    Implementation of the extended filtering algorithm, as defined in point
    3.3.2, of U{RFC 4647<http://www.rfc-editor.org/rfc/rfc4647.txt>}, on
    matching language ranges and language tags.
    Needed to handle the C{rdf:PlainLiteral} datatype.
    @param range: language range
    @param lang: language tag
    @rtype: boolean

        @author: U{Ivan Herman<a href="http://www.w3.org/People/Ivan/">}

        Taken from `RDFClosure/RestrictedDatatype.py`__

    .. __:http://dev.w3.org/2004/PythonLib-IH/RDFClosure/RestrictedDatatype.py

    """

    def _match(r: str, l_: str) -> bool:
        """
        Matching of a range and language item: either range is a wildcard
        or the two are equal
        @param r: language range item
        @param l_: language tag item
        @rtype: boolean
        """
        return r == '*' or r == l_
    rangeList = range.strip().lower().split('-')
    langList = lang.strip().lower().split('-')
    if not _match(rangeList[0], langList[0]):
        return False
    if len(rangeList) > len(langList):
        return False
    return all((_match(*x) for x in zip(rangeList, langList)))