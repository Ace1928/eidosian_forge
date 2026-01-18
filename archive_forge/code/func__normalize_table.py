from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
def _normalize_table(self, table: exp.Table | str, dialect: DialectType=None, normalize: t.Optional[bool]=None) -> exp.Table:
    dialect = dialect or self.dialect
    normalize = self.normalize if normalize is None else normalize
    normalized_table = exp.maybe_parse(table, into=exp.Table, dialect=dialect, copy=normalize)
    if normalize:
        for arg in exp.TABLE_PARTS:
            value = normalized_table.args.get(arg)
            if isinstance(value, exp.Identifier):
                normalized_table.set(arg, normalize_name(value, dialect=dialect, is_table=True, normalize=normalize))
    return normalized_table