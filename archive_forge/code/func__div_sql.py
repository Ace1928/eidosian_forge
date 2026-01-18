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
def _div_sql(self: generator.Generator, e: exp.Div) -> str:
    denominator = self.sql(e, 'expression')
    if e.args.get('safe'):
        denominator += ' or None'
    sql = f'DIV({self.sql(e, 'this')}, {denominator})'
    if e.args.get('typed'):
        sql = f'int({sql})'
    return sql