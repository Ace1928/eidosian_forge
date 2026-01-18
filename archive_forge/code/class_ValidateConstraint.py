from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
class ValidateConstraint(Operation):
    """Validate a table NOT VALID constraint."""

    def __init__(self, model_name, name):
        self.model_name = model_name
        self.name = name

    def describe(self):
        return 'Validate constraint %s on model %s' % (self.name, self.model_name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.execute('ALTER TABLE %s VALIDATE CONSTRAINT %s' % (schema_editor.quote_name(model._meta.db_table), schema_editor.quote_name(self.name)))

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def state_forwards(self, app_label, state):
        pass

    @property
    def migration_name_fragment(self):
        return '%s_validate_%s' % (self.model_name.lower(), self.name.lower())

    def deconstruct(self):
        return (self.__class__.__name__, [], {'model_name': self.model_name, 'name': self.name})