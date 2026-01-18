from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
class AddField(FieldOperation):
    """Add a field to a model."""

    def __init__(self, model_name, name, field, preserve_default=True):
        self.preserve_default = preserve_default
        super().__init__(model_name, name, field)

    def deconstruct(self):
        kwargs = {'model_name': self.model_name, 'name': self.name, 'field': self.field}
        if self.preserve_default is not True:
            kwargs['preserve_default'] = self.preserve_default
        return (self.__class__.__name__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.add_field(app_label, self.model_name_lower, self.name, self.field, self.preserve_default)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            field = to_model._meta.get_field(self.name)
            if not self.preserve_default:
                field.default = self.field.default
            schema_editor.add_field(from_model, field)
            if not self.preserve_default:
                field.default = NOT_PROVIDED

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        from_model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, from_model):
            schema_editor.remove_field(from_model, from_model._meta.get_field(self.name))

    def describe(self):
        return 'Add field %s to %s' % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return '%s_%s' % (self.model_name_lower, self.name_lower)

    def reduce(self, operation, app_label):
        if isinstance(operation, FieldOperation) and self.is_same_field_operation(operation):
            if isinstance(operation, AlterField):
                return [AddField(model_name=self.model_name, name=operation.name, field=operation.field)]
            elif isinstance(operation, RemoveField):
                return []
            elif isinstance(operation, RenameField):
                return [AddField(model_name=self.model_name, name=operation.new_name, field=self.field)]
        return super().reduce(operation, app_label)