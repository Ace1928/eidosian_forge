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
class SchemaDropper(InvokeDropDDLBase):

    def __init__(self, dialect, connection, checkfirst=False, tables=None, **kwargs):
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst
        self.tables = tables
        self.preparer = dialect.identifier_preparer
        self.dialect = dialect
        self.memo = {}

    def visit_metadata(self, metadata):
        if self.tables is not None:
            tables = self.tables
        else:
            tables = list(metadata.tables.values())
        try:
            unsorted_tables = [t for t in tables if self._can_drop_table(t)]
            collection = list(reversed(sort_tables_and_constraints(unsorted_tables, filter_fn=lambda constraint: False if not self.dialect.supports_alter or constraint.name is None else None)))
        except exc.CircularDependencyError as err2:
            if not self.dialect.supports_alter:
                util.warn("Can't sort tables for DROP; an unresolvable foreign key dependency exists between tables: %s; and backend does not support ALTER.  To restore at least a partial sort, apply use_alter=True to ForeignKey and ForeignKeyConstraint objects involved in the cycle to mark these as known cycles that will be ignored." % ', '.join(sorted([t.fullname for t in err2.cycles])))
                collection = [(t, ()) for t in unsorted_tables]
            else:
                raise exc.CircularDependencyError(err2.args[0], err2.cycles, err2.edges, msg="Can't sort tables for DROP; an unresolvable foreign key dependency exists between tables: %s.  Please ensure that the ForeignKey and ForeignKeyConstraint objects involved in the cycle have names so that they can be dropped using DROP CONSTRAINT." % ', '.join(sorted([t.fullname for t in err2.cycles]))) from err2
        seq_coll = [s for s in metadata._sequences.values() if self._can_drop_sequence(s)]
        event_collection = [t for t, fks in collection if t is not None]
        with self.with_ddl_events(metadata, tables=event_collection, checkfirst=self.checkfirst):
            for table, fkcs in collection:
                if table is not None:
                    self.traverse_single(table, drop_ok=True, _is_metadata_operation=True, _ignore_sequences=seq_coll)
                else:
                    for fkc in fkcs:
                        self.traverse_single(fkc)
            for seq in seq_coll:
                self.traverse_single(seq, drop_ok=seq.column is None)

    def _can_drop_table(self, table):
        self.dialect.validate_identifier(table.name)
        effective_schema = self.connection.schema_for_object(table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or self.dialect.has_table(self.connection, table.name, schema=effective_schema)

    def _can_drop_index(self, index):
        effective_schema = self.connection.schema_for_object(index.table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or self.dialect.has_index(self.connection, index.table.name, index.name, schema=effective_schema)

    def _can_drop_sequence(self, sequence):
        effective_schema = self.connection.schema_for_object(sequence)
        return self.dialect.supports_sequences and ((not self.dialect.sequences_optional or not sequence.optional) and (not self.checkfirst or self.dialect.has_sequence(self.connection, sequence.name, schema=effective_schema)))

    def visit_index(self, index, drop_ok=False):
        if not drop_ok and (not self._can_drop_index(index)):
            return
        with self.with_ddl_events(index):
            DropIndex(index)(index, self.connection)

    def visit_table(self, table, drop_ok=False, _is_metadata_operation=False, _ignore_sequences=()):
        if not drop_ok and (not self._can_drop_table(table)):
            return
        with self.with_ddl_events(table, checkfirst=self.checkfirst, _is_metadata_operation=_is_metadata_operation):
            DropTable(table)._invoke_with(self.connection)
            for column in table.columns:
                if column.default is not None and column.default not in _ignore_sequences:
                    self.traverse_single(column.default)

    def visit_foreign_key_constraint(self, constraint):
        if not self.dialect.supports_alter:
            return
        with self.with_ddl_events(constraint):
            DropConstraint(constraint)._invoke_with(self.connection)

    def visit_sequence(self, sequence, drop_ok=False):
        if not drop_ok and (not self._can_drop_sequence(sequence)):
            return
        with self.with_ddl_events(sequence):
            DropSequence(sequence)._invoke_with(self.connection)