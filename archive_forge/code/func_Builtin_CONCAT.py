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
def Builtin_CONCAT(expr: Expr, ctx) -> Literal:
    """
    http://www.w3.org/TR/sparql11-query/#func-concat
    """
    dt = set((x.datatype for x in expr.arg if isinstance(x, Literal)))
    dt = dt.pop() if len(dt) == 1 else None
    lang = set((x.language for x in expr.arg if isinstance(x, Literal)))
    lang = lang.pop() if len(lang) == 1 else None
    return Literal(''.join((string(x) for x in expr.arg)), datatype=dt, lang=lang)