from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class AddConstraint(IndexOperation):
    option_name = 'constraints'

    def __init__(self, model_name, constraint):
        self.model_name = model_name
        self.constraint = constraint

    def state_forwards(self, app_label, state):
        state.add_constraint(app_label, self.model_name_lower, self.constraint)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.add_constraint(model, self.constraint)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.remove_constraint(model, self.constraint)

    def deconstruct(self):
        return (self.__class__.__name__, [], {'model_name': self.model_name, 'constraint': self.constraint})

    def describe(self):
        return 'Create constraint %s on model %s' % (self.constraint.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return '%s_%s' % (self.model_name_lower, self.constraint.name.lower())

    def reduce(self, operation, app_label):
        if isinstance(operation, RemoveConstraint) and self.model_name_lower == operation.model_name_lower and (self.constraint.name == operation.name):
            return []
        return super().reduce(operation, app_label)