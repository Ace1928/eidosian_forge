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
def _lambda_sql(self, e: exp.Lambda) -> str:
    names = {e.name.lower() for e in e.expressions}
    e = e.transform(lambda n: exp.var(n.name) if isinstance(n, exp.Identifier) and n.name.lower() in names else n).assert_is(exp.Lambda)
    return f'lambda {self.expressions(e, flat=True)}: {self.sql(e, 'this')}'