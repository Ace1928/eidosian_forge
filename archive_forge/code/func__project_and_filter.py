import ast
import collections
import itertools
import math
from sqlglot import exp, generator, planner, tokens
from sqlglot.dialects.dialect import Dialect, inline_array_sql
from sqlglot.errors import ExecuteError
from sqlglot.executor.context import Context
from sqlglot.executor.env import ENV
from sqlglot.executor.table import RowReader, Table
from sqlglot.helper import csv_reader, ensure_list, subclasses
def _project_and_filter(self, context, step, table_iter):
    sink = self.table(step.projections if step.projections else context.columns)
    condition = self.generate(step.condition)
    projections = self.generate_tuple(step.projections)
    for reader in table_iter:
        if len(sink) >= step.limit:
            break
        if condition and (not context.eval(condition)):
            continue
        if projections:
            sink.append(context.eval_tuple(projections))
        else:
            sink.append(reader.row)
    return sink