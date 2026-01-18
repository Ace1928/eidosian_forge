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
def _case_sql(self, expression):
    this = self.sql(expression, 'this')
    chain = self.sql(expression, 'default') or 'None'
    for e in reversed(expression.args['ifs']):
        true = self.sql(e, 'true')
        condition = self.sql(e, 'this')
        condition = f'{this} = ({condition})' if this else condition
        chain = f'{true} if {condition} else ({chain})'
    return chain