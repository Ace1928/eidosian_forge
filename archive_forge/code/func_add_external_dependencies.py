import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def add_external_dependencies(self, key, migration):
    for parent in migration.dependencies:
        if key[0] == parent[0]:
            continue
        parent = self.check_key(parent, key[0])
        if parent is not None:
            self.graph.add_dependency(migration, key, parent, skip_validation=True)
    for child in migration.run_before:
        child = self.check_key(child, key[0])
        if child is not None:
            self.graph.add_dependency(migration, child, key, skip_validation=True)