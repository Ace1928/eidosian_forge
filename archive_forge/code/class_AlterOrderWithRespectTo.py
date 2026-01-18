from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class AlterOrderWithRespectTo(ModelOptionOperation):
    """Represent a change with the order_with_respect_to option."""
    option_name = 'order_with_respect_to'

    def __init__(self, name, order_with_respect_to):
        self.order_with_respect_to = order_with_respect_to
        super().__init__(name)

    def deconstruct(self):
        kwargs = {'name': self.name, 'order_with_respect_to': self.order_with_respect_to}
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(app_label, self.name_lower, {self.option_name: self.order_with_respect_to})

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.name)
            if from_model._meta.order_with_respect_to and (not to_model._meta.order_with_respect_to):
                schema_editor.remove_field(from_model, from_model._meta.get_field('_order'))
            elif to_model._meta.order_with_respect_to and (not from_model._meta.order_with_respect_to):
                field = to_model._meta.get_field('_order')
                if not field.has_default():
                    field.default = 0
                schema_editor.add_field(from_model, field)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.database_forwards(app_label, schema_editor, from_state, to_state)

    def references_field(self, model_name, name, app_label):
        return self.references_model(model_name, app_label) and (self.order_with_respect_to is None or name == self.order_with_respect_to)

    def describe(self):
        return 'Set order_with_respect_to on %s to %s' % (self.name, self.order_with_respect_to)

    @property
    def migration_name_fragment(self):
        return 'alter_%s_order_with_respect_to' % self.name_lower