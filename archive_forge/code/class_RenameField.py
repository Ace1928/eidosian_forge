from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
class RenameField(FieldOperation):
    """Rename a field on the model. Might affect db_column too."""

    def __init__(self, model_name, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        super().__init__(model_name, old_name)

    @cached_property
    def old_name_lower(self):
        return self.old_name.lower()

    @cached_property
    def new_name_lower(self):
        return self.new_name.lower()

    def deconstruct(self):
        kwargs = {'model_name': self.model_name, 'old_name': self.old_name, 'new_name': self.new_name}
        return (self.__class__.__name__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.rename_field(app_label, self.model_name_lower, self.old_name, self.new_name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            schema_editor.alter_field(from_model, from_model._meta.get_field(self.old_name), to_model._meta.get_field(self.new_name))

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            schema_editor.alter_field(from_model, from_model._meta.get_field(self.new_name), to_model._meta.get_field(self.old_name))

    def describe(self):
        return 'Rename field %s on %s to %s' % (self.old_name, self.model_name, self.new_name)

    @property
    def migration_name_fragment(self):
        return 'rename_%s_%s_%s' % (self.old_name_lower, self.model_name_lower, self.new_name_lower)

    def references_field(self, model_name, name, app_label):
        return self.references_model(model_name, app_label) and (name.lower() == self.old_name_lower or name.lower() == self.new_name_lower)

    def reduce(self, operation, app_label):
        if isinstance(operation, RenameField) and self.is_same_model_operation(operation) and (self.new_name_lower == operation.old_name_lower):
            return [RenameField(self.model_name, self.old_name, operation.new_name)]
        return super(FieldOperation, self).reduce(operation, app_label) or not (operation.references_field(self.model_name, self.old_name, app_label) or operation.references_field(self.model_name, self.new_name, app_label))