import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
class IntervalVar:
    """Represents an Interval variable.

    An interval variable is both a constraint and a variable. It is defined by
    three integer variables: start, size, and end.

    It is a constraint because, internally, it enforces that start + size == end.

    It is also a variable as it can appear in specific scheduling constraints:
    NoOverlap, NoOverlap2D, Cumulative.

    Optionally, an enforcement literal can be added to this constraint, in which
    case these scheduling constraints will ignore interval variables with
    enforcement literals assigned to false. Conversely, these constraints will
    also set these enforcement literals to false if they cannot fit these
    intervals into the schedule.
    """

    def __init__(self, model: cp_model_pb2.CpModelProto, start: Union[cp_model_pb2.LinearExpressionProto, int], size: Optional[cp_model_pb2.LinearExpressionProto], end: Optional[cp_model_pb2.LinearExpressionProto], is_present_index: Optional[int], name: Optional[str]):
        self.__model: cp_model_pb2.CpModelProto = model
        if size is None and end is None and (is_present_index is None) and (name is None):
            self.__index: int = cast(int, start)
            self.__ct: cp_model_pb2.ConstraintProto = model.constraints[self.__index]
        else:
            self.__index: int = len(model.constraints)
            self.__ct: cp_model_pb2.ConstraintProto = self.__model.constraints.add()
            self.__ct.interval.start.CopyFrom(start)
            self.__ct.interval.size.CopyFrom(size)
            self.__ct.interval.end.CopyFrom(end)
            if is_present_index is not None:
                self.__ct.enforcement_literal.append(is_present_index)
            if name:
                self.__ct.name = name

    @property
    def index(self) -> int:
        """Returns the index of the interval constraint in the model."""
        return self.__index

    @property
    def proto(self) -> cp_model_pb2.IntervalConstraintProto:
        """Returns the interval protobuf."""
        return self.__ct.interval

    def __str__(self):
        return self.__ct.name

    def __repr__(self):
        interval = self.__ct.interval
        if self.__ct.enforcement_literal:
            return '%s(start = %s, size = %s, end = %s, is_present = %s)' % (self.__ct.name, short_expr_name(self.__model, interval.start), short_expr_name(self.__model, interval.size), short_expr_name(self.__model, interval.end), short_name(self.__model, self.__ct.enforcement_literal[0]))
        else:
            return '%s(start = %s, size = %s, end = %s)' % (self.__ct.name, short_expr_name(self.__model, interval.start), short_expr_name(self.__model, interval.size), short_expr_name(self.__model, interval.end))

    @property
    def name(self) -> str:
        if not self.__ct or not self.__ct.name:
            return ''
        return self.__ct.name

    def start_expr(self) -> LinearExprT:
        return LinearExpr.rebuild_from_linear_expression_proto(self.__model, self.__ct.interval.start)

    def size_expr(self) -> LinearExprT:
        return LinearExpr.rebuild_from_linear_expression_proto(self.__model, self.__ct.interval.size)

    def end_expr(self) -> LinearExprT:
        return LinearExpr.rebuild_from_linear_expression_proto(self.__model, self.__ct.interval.end)

    def Name(self) -> str:
        return self.name

    def Index(self) -> int:
        return self.index

    def Proto(self) -> cp_model_pb2.IntervalConstraintProto:
        return self.proto
    StartExpr = start_expr
    SizeExpr = size_expr
    EndExpr = end_expr