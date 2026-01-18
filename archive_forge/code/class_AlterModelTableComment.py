from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class AlterModelTableComment(ModelOptionOperation):

    def __init__(self, name, table_comment):
        self.table_comment = table_comment
        super().__init__(name)

    def deconstruct(self):
        kwargs = {'name': self.name, 'table_comment': self.table_comment}
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(app_label, self.name_lower, {'db_table_comment': self.table_comment})

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.name)
            schema_editor.alter_db_table_comment(new_model, old_model._meta.db_table_comment, new_model._meta.db_table_comment)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        return self.database_forwards(app_label, schema_editor, from_state, to_state)

    def describe(self):
        return f'Alter {self.name} table comment'

    @property
    def migration_name_fragment(self):
        return f'alter_{self.name_lower}_table_comment'