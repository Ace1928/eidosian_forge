from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
class CreateExtension(Operation):
    reversible = True

    def __init__(self, name):
        self.name = name

    def state_forwards(self, app_label, state):
        pass

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        if schema_editor.connection.vendor != 'postgresql' or not router.allow_migrate(schema_editor.connection.alias, app_label):
            return
        if not self.extension_exists(schema_editor, self.name):
            schema_editor.execute('CREATE EXTENSION IF NOT EXISTS %s' % schema_editor.quote_name(self.name))
        get_hstore_oids.cache_clear()
        get_citext_oids.cache_clear()
        register_type_handlers(schema_editor.connection)
        if hasattr(schema_editor.connection, 'register_geometry_adapters'):
            schema_editor.connection.register_geometry_adapters(schema_editor.connection.connection, True)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if not router.allow_migrate(schema_editor.connection.alias, app_label):
            return
        if self.extension_exists(schema_editor, self.name):
            schema_editor.execute('DROP EXTENSION IF EXISTS %s' % schema_editor.quote_name(self.name))
        get_hstore_oids.cache_clear()
        get_citext_oids.cache_clear()

    def extension_exists(self, schema_editor, extension):
        with schema_editor.connection.cursor() as cursor:
            cursor.execute('SELECT 1 FROM pg_extension WHERE extname = %s', [extension])
            return bool(cursor.fetchone())

    def describe(self):
        return 'Creates extension %s' % self.name

    @property
    def migration_name_fragment(self):
        return 'create_extension_%s' % self.name