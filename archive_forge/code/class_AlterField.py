from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
class AlterField(FieldOperation):
    """
    Alter a field's database column (e.g. null, max_length) to the provided
    new field.
    """

    def __init__(self, model_name, name, field, preserve_default=True):
        self.preserve_default = preserve_default
        super().__init__(model_name, name, field)

    def deconstruct(self):
        kwargs = {'model_name': self.model_name, 'name': self.name, 'field': self.field}
        if self.preserve_default is not True:
            kwargs['preserve_default'] = self.preserve_default
        return (self.__class__.__name__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_field(app_label, self.model_name_lower, self.name, self.field, self.preserve_default)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            from_field = from_model._meta.get_field(self.name)
            to_field = to_model._meta.get_field(self.name)
            if not self.preserve_default:
                to_field.default = self.field.default
            schema_editor.alter_field(from_model, from_field, to_field)
            if not self.preserve_default:
                to_field.default = NOT_PROVIDED

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.database_forwards(app_label, schema_editor, from_state, to_state)

    def describe(self):
        return 'Alter field %s on %s' % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return 'alter_%s_%s' % (self.model_name_lower, self.name_lower)

    def reduce(self, operation, app_label):
        if isinstance(operation, (AlterField, RemoveField)) and self.is_same_field_operation(operation):
            return [operation]
        elif isinstance(operation, RenameField) and self.is_same_field_operation(operation) and (self.field.db_column is None):
            return [operation, AlterField(model_name=self.model_name, name=operation.new_name, field=self.field)]
        return super().reduce(operation, app_label)