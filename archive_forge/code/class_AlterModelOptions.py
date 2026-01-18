from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class AlterModelOptions(ModelOptionOperation):
    """
    Set new model options that don't directly affect the database schema
    (like verbose_name, permissions, ordering). Python code in migrations
    may still need them.
    """
    ALTER_OPTION_KEYS = ['base_manager_name', 'default_manager_name', 'default_related_name', 'get_latest_by', 'managed', 'ordering', 'permissions', 'default_permissions', 'select_on_save', 'verbose_name', 'verbose_name_plural']

    def __init__(self, name, options):
        self.options = options
        super().__init__(name)

    def deconstruct(self):
        kwargs = {'name': self.name, 'options': self.options}
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(app_label, self.name_lower, self.options, self.ALTER_OPTION_KEYS)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def describe(self):
        return 'Change Meta options on %s' % self.name

    @property
    def migration_name_fragment(self):
        return 'alter_%s_options' % self.name_lower