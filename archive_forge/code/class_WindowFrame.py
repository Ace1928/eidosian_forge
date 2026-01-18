import copy
import datetime
import functools
import inspect
from collections import defaultdict
from decimal import Decimal
from types import NoneType
from uuid import UUID
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class WindowFrame(Expression):
    """
    Model the frame clause in window expressions. There are two types of frame
    clauses which are subclasses, however, all processing and validation (by no
    means intended to be complete) is done here. Thus, providing an end for a
    frame is optional (the default is UNBOUNDED FOLLOWING, which is the last
    row in the frame).
    """
    template = '%(frame_type)s BETWEEN %(start)s AND %(end)s'

    def __init__(self, start=None, end=None):
        self.start = Value(start)
        self.end = Value(end)

    def set_source_expressions(self, exprs):
        self.start, self.end = exprs

    def get_source_expressions(self):
        return [self.start, self.end]

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        start, end = self.window_frame_start_end(connection, self.start.value, self.end.value)
        return (self.template % {'frame_type': self.frame_type, 'start': start, 'end': end}, [])

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def get_group_by_cols(self):
        return []

    def __str__(self):
        if self.start.value is not None and self.start.value < 0:
            start = '%d %s' % (abs(self.start.value), connection.ops.PRECEDING)
        elif self.start.value is not None and self.start.value == 0:
            start = connection.ops.CURRENT_ROW
        else:
            start = connection.ops.UNBOUNDED_PRECEDING
        if self.end.value is not None and self.end.value > 0:
            end = '%d %s' % (self.end.value, connection.ops.FOLLOWING)
        elif self.end.value is not None and self.end.value == 0:
            end = connection.ops.CURRENT_ROW
        else:
            end = connection.ops.UNBOUNDED_FOLLOWING
        return self.template % {'frame_type': self.frame_type, 'start': start, 'end': end}

    def window_frame_start_end(self, connection, start, end):
        raise NotImplementedError('Subclasses must implement window_frame_start_end().')