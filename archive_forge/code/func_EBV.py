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
def EBV(rt: Union[Identifier, SPARQLError, Expr]) -> bool:
    """
    Effective Boolean Value (EBV)

    * If the argument is a typed literal with a datatype of xsd:boolean,
      the EBV is the value of that argument.
    * If the argument is a plain literal or a typed literal with a
      datatype of xsd:string, the EBV is false if the operand value
      has zero length; otherwise the EBV is true.
    * If the argument is a numeric type or a typed literal with a datatype
      derived from a numeric type, the EBV is false if the operand value is
      NaN or is numerically equal to zero; otherwise the EBV is true.
    * All other arguments, including unbound arguments, produce a type error.

    """
    if isinstance(rt, Literal):
        if rt.datatype == XSD.boolean:
            return rt.toPython()
        elif rt.datatype == XSD.string or rt.datatype is None:
            return len(rt) > 0
        else:
            pyRT = rt.toPython()
            if isinstance(pyRT, Literal):
                raise SPARQLTypeError("http://www.w3.org/TR/rdf-sparql-query/#ebv - ' +                     'Could not determine the EBV for : %r" % rt)
            else:
                return bool(pyRT)
    else:
        raise SPARQLTypeError("http://www.w3.org/TR/rdf-sparql-query/#ebv - ' +             'Only literals have Boolean values! %r" % rt)