from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class AddIndex(IndexOperation):
    """Add an index on a model."""

    def __init__(self, model_name, index):
        self.model_name = model_name
        if not index.name:
            raise ValueError("Indexes passed to AddIndex operations require a name argument. %r doesn't have one." % index)
        self.index = index

    def state_forwards(self, app_label, state):
        state.add_index(app_label, self.model_name_lower, self.index)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.add_index(model, self.index)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.remove_index(model, self.index)

    def deconstruct(self):
        kwargs = {'model_name': self.model_name, 'index': self.index}
        return (self.__class__.__qualname__, [], kwargs)

    def describe(self):
        if self.index.expressions:
            return 'Create index %s on %s on model %s' % (self.index.name, ', '.join([str(expression) for expression in self.index.expressions]), self.model_name)
        return 'Create index %s on field(s) %s of model %s' % (self.index.name, ', '.join(self.index.fields), self.model_name)

    @property
    def migration_name_fragment(self):
        return '%s_%s' % (self.model_name_lower, self.index.name.lower())

    def reduce(self, operation, app_label):
        if isinstance(operation, RemoveIndex) and self.index.name == operation.name:
            return []
        if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
            self.index.name = operation.new_name
            return [AddIndex(model_name=self.model_name, index=self.index)]
        return super().reduce(operation, app_label)