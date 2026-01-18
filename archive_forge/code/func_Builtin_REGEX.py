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
def Builtin_REGEX(expr: Expr, ctx) -> Literal:
    """
    http://www.w3.org/TR/sparql11-query/#func-regex
    Invokes the XPath fn:matches function to match text against a regular
    expression pattern.
    The regular expression language is defined in XQuery 1.0 and XPath 2.0
    Functions and Operators section 7.6.1 Regular Expression Syntax
    """
    text = string(expr.text)
    pattern = string(expr.pattern)
    flags = expr.flags
    cFlag = 0
    if flags:
        flagMap = dict([('i', re.IGNORECASE), ('s', re.DOTALL), ('m', re.MULTILINE)])
        cFlag = reduce(pyop.or_, [flagMap.get(f, 0) for f in flags])
    return Literal(bool(re.search(str(pattern), text, cFlag)))