from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
class RemoveCollation(CollationOperation):
    """Remove a collation."""

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        if schema_editor.connection.vendor != 'postgresql' or not router.allow_migrate(schema_editor.connection.alias, app_label):
            return
        self.remove_collation(schema_editor)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if not router.allow_migrate(schema_editor.connection.alias, app_label):
            return
        self.create_collation(schema_editor)

    def describe(self):
        return f'Remove collation {self.name}'

    @property
    def migration_name_fragment(self):
        return 'remove_collation_%s' % self.name.lower()