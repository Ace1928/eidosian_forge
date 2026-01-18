from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def _add_except_columns(expression: exp.Expression, tables, except_columns: t.Dict[int, t.Set[str]]) -> None:
    except_ = expression.args.get('except')
    if not except_:
        return
    columns = {e.name for e in except_}
    for table in tables:
        except_columns[id(table)] = columns