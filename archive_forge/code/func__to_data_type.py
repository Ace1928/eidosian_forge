from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
def _to_data_type(self, schema_type: str, dialect: DialectType=None) -> exp.DataType:
    """
        Convert a type represented as a string to the corresponding `sqlglot.exp.DataType` object.

        Args:
            schema_type: the type we want to convert.
            dialect: the SQL dialect that will be used to parse `schema_type`, if needed.

        Returns:
            The resulting expression type.
        """
    if schema_type not in self._type_mapping_cache:
        dialect = dialect or self.dialect
        udt = Dialect.get_or_raise(dialect).SUPPORTS_USER_DEFINED_TYPES
        try:
            expression = exp.DataType.build(schema_type, dialect=dialect, udt=udt)
            self._type_mapping_cache[schema_type] = expression
        except AttributeError:
            in_dialect = f' in dialect {dialect}' if dialect else ''
            raise SchemaError(f"Failed to build type '{schema_type}'{in_dialect}.")
    return self._type_mapping_cache[schema_type]