from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
class SchemaGenerator(InvokeCreateDDLBase):

    def __init__(self, dialect, connection, checkfirst=False, tables=None, **kwargs):
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst
        self.tables = tables
        self.preparer = dialect.identifier_preparer
        self.dialect = dialect
        self.memo = {}

    def _can_create_table(self, table):
        self.dialect.validate_identifier(table.name)
        effective_schema = self.connection.schema_for_object(table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or not self.dialect.has_table(self.connection, table.name, schema=effective_schema)

    def _can_create_index(self, index):
        effective_schema = self.connection.schema_for_object(index.table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or not self.dialect.has_index(self.connection, index.table.name, index.name, schema=effective_schema)

    def _can_create_sequence(self, sequence):
        effective_schema = self.connection.schema_for_object(sequence)
        return self.dialect.supports_sequences and ((not self.dialect.sequences_optional or not sequence.optional) and (not self.checkfirst or not self.dialect.has_sequence(self.connection, sequence.name, schema=effective_schema)))

    def visit_metadata(self, metadata):
        if self.tables is not None:
            tables = self.tables
        else:
            tables = list(metadata.tables.values())
        collection = sort_tables_and_constraints([t for t in tables if self._can_create_table(t)])
        seq_coll = [s for s in metadata._sequences.values() if s.column is None and self._can_create_sequence(s)]
        event_collection = [t for t, fks in collection if t is not None]
        with self.with_ddl_events(metadata, tables=event_collection, checkfirst=self.checkfirst):
            for seq in seq_coll:
                self.traverse_single(seq, create_ok=True)
            for table, fkcs in collection:
                if table is not None:
                    self.traverse_single(table, create_ok=True, include_foreign_key_constraints=fkcs, _is_metadata_operation=True)
                else:
                    for fkc in fkcs:
                        self.traverse_single(fkc)

    def visit_table(self, table, create_ok=False, include_foreign_key_constraints=None, _is_metadata_operation=False):
        if not create_ok and (not self._can_create_table(table)):
            return
        with self.with_ddl_events(table, checkfirst=self.checkfirst, _is_metadata_operation=_is_metadata_operation):
            for column in table.columns:
                if column.default is not None:
                    self.traverse_single(column.default)
            if not self.dialect.supports_alter:
                include_foreign_key_constraints = None
            CreateTable(table, include_foreign_key_constraints=include_foreign_key_constraints)._invoke_with(self.connection)
            if hasattr(table, 'indexes'):
                for index in table.indexes:
                    self.traverse_single(index, create_ok=True)
            if self.dialect.supports_comments and (not self.dialect.inline_comments):
                if table.comment is not None:
                    SetTableComment(table)._invoke_with(self.connection)
                for column in table.columns:
                    if column.comment is not None:
                        SetColumnComment(column)._invoke_with(self.connection)
                if self.dialect.supports_constraint_comments:
                    for constraint in table.constraints:
                        if constraint.comment is not None:
                            self.connection.execute(SetConstraintComment(constraint))

    def visit_foreign_key_constraint(self, constraint):
        if not self.dialect.supports_alter:
            return
        with self.with_ddl_events(constraint):
            AddConstraint(constraint)._invoke_with(self.connection)

    def visit_sequence(self, sequence, create_ok=False):
        if not create_ok and (not self._can_create_sequence(sequence)):
            return
        with self.with_ddl_events(sequence):
            CreateSequence(sequence)._invoke_with(self.connection)

    def visit_index(self, index, create_ok=False):
        if not create_ok and (not self._can_create_index(index)):
            return
        with self.with_ddl_events(index):
            CreateIndex(index)._invoke_with(self.connection)