import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node
@classmethod
def _gen_cols(cls, exprs, include_external=False, resolve_refs=True):
    for expr in exprs:
        if isinstance(expr, Col):
            yield expr
        elif include_external and callable(getattr(expr, 'get_external_cols', None)):
            yield from expr.get_external_cols()
        elif hasattr(expr, 'get_source_expressions'):
            if not resolve_refs and isinstance(expr, Ref):
                continue
            yield from cls._gen_cols(expr.get_source_expressions(), include_external=include_external, resolve_refs=resolve_refs)