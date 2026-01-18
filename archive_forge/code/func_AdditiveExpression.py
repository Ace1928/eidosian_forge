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
def AdditiveExpression(e: Expr, ctx: Union[QueryContext, FrozenBindings]) -> Literal:
    expr = e.expr
    other = e.other
    if other is None:
        return expr
    if hasattr(expr, 'datatype') and (expr.datatype in XSD_DateTime_DTs or expr.datatype in XSD_Duration_DTs):
        res = dateTimeObjects(expr)
        dt = expr.datatype
        for op, term in zip(e.op, other):
            if dt in XSD_DateTime_DTs and dt == term.datatype and (op == '-'):
                if len(other) > 1:
                    error_message = "Can't evaluate multiple %r arguments"
                    raise SPARQLError(error_message, dt.datatype)
                else:
                    n = dateTimeObjects(term)
                    res = calculateDuration(res, n)
                    return res
            elif dt in XSD_DateTime_DTs and term.datatype in XSD_Duration_DTs:
                n = dateTimeObjects(term)
                res = calculateFinalDateTime(res, dt, n, term.datatype, op)
                return res
            elif dt in XSD_Duration_DTs and term.datatype in XSD_DateTime_DTs:
                if op == '+':
                    n = dateTimeObjects(term)
                    res = calculateFinalDateTime(res, dt, n, term.datatype, op)
                    return res
            else:
                raise SPARQLError('Invalid DateTime Operations')
    else:
        res = numeric(expr)
        dt = expr.datatype
        for op, term in zip(e.op, other):
            n = numeric(term)
            if isinstance(n, Decimal) and isinstance(res, float):
                n = float(n)
            if isinstance(n, float) and isinstance(res, Decimal):
                res = float(res)
            dt = type_promotion(dt, term.datatype)
            if op == '+':
                res += n
            else:
                res -= n
        return Literal(res, datatype=dt)