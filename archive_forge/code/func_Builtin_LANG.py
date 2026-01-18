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
def Builtin_LANG(e: Expr, ctx) -> Literal:
    """
    http://www.w3.org/TR/sparql11-query/#func-lang

    Returns the language tag of ltrl, if it has one. It returns "" if ltrl has
    no language tag. Note that the RDF data model does not include literals
    with an empty language tag.
    """
    l_ = literal(e.arg)
    return Literal(l_.language or '')