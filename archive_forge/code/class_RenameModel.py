from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class RenameModel(ModelOperation):
    """Rename a model."""

    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        super().__init__(old_name)

    @cached_property
    def old_name_lower(self):
        return self.old_name.lower()

    @cached_property
    def new_name_lower(self):
        return self.new_name.lower()

    def deconstruct(self):
        kwargs = {'old_name': self.old_name, 'new_name': self.new_name}
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.rename_model(app_label, self.old_name, self.new_name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.new_name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.old_name)
            schema_editor.alter_db_table(new_model, old_model._meta.db_table, new_model._meta.db_table)
            for related_object in old_model._meta.related_objects:
                if related_object.related_model == old_model:
                    model = new_model
                    related_key = (app_label, self.new_name_lower)
                else:
                    model = related_object.related_model
                    related_key = (related_object.related_model._meta.app_label, related_object.related_model._meta.model_name)
                to_field = to_state.apps.get_model(*related_key)._meta.get_field(related_object.field.name)
                schema_editor.alter_field(model, related_object.field, to_field)
            fields = zip(old_model._meta.local_many_to_many, new_model._meta.local_many_to_many)
            for old_field, new_field in fields:
                if new_field.model == new_field.related_model or not new_field.remote_field.through._meta.auto_created:
                    continue
                schema_editor._alter_many_to_many(new_model, old_field, new_field, strict=False)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.new_name_lower, self.old_name_lower = (self.old_name_lower, self.new_name_lower)
        self.new_name, self.old_name = (self.old_name, self.new_name)
        self.database_forwards(app_label, schema_editor, from_state, to_state)
        self.new_name_lower, self.old_name_lower = (self.old_name_lower, self.new_name_lower)
        self.new_name, self.old_name = (self.old_name, self.new_name)

    def references_model(self, name, app_label):
        return name.lower() == self.old_name_lower or name.lower() == self.new_name_lower

    def describe(self):
        return 'Rename model %s to %s' % (self.old_name, self.new_name)

    @property
    def migration_name_fragment(self):
        return 'rename_%s_%s' % (self.old_name_lower, self.new_name_lower)

    def reduce(self, operation, app_label):
        if isinstance(operation, RenameModel) and self.new_name_lower == operation.old_name_lower:
            return [RenameModel(self.old_name, operation.new_name)]
        return super(ModelOperation, self).reduce(operation, app_label) or not operation.references_model(self.new_name, app_label)