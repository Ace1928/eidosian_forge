from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class AlterModelManagers(ModelOptionOperation):
    """Alter the model's managers."""
    serialization_expand_args = ['managers']

    def __init__(self, name, managers):
        self.managers = managers
        super().__init__(name)

    def deconstruct(self):
        return (self.__class__.__qualname__, [self.name, self.managers], {})

    def state_forwards(self, app_label, state):
        state.alter_model_managers(app_label, self.name_lower, self.managers)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def describe(self):
        return 'Change managers on %s' % self.name

    @property
    def migration_name_fragment(self):
        return 'alter_%s_managers' % self.name_lower