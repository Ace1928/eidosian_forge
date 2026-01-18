from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
class AddConstraintNotValid(AddConstraint):
    """
    Add a table constraint without enforcing validation, using PostgreSQL's
    NOT VALID syntax.
    """

    def __init__(self, model_name, constraint):
        if not isinstance(constraint, CheckConstraint):
            raise TypeError('AddConstraintNotValid.constraint must be a check constraint.')
        super().__init__(model_name, constraint)

    def describe(self):
        return 'Create not valid constraint %s on model %s' % (self.constraint.name, self.model_name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            constraint_sql = self.constraint.create_sql(model, schema_editor)
            if constraint_sql:
                schema_editor.execute(str(constraint_sql) + ' NOT VALID', params=None)

    @property
    def migration_name_fragment(self):
        return super().migration_name_fragment + '_not_valid'