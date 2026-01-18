import sys
from django.apps import apps
from django.core.management.base import BaseCommand
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.recorder import MigrationRecorder
def _validate_app_names(self, loader, app_names):
    has_bad_names = False
    for app_name in app_names:
        try:
            apps.get_app_config(app_name)
        except LookupError as err:
            self.stderr.write(str(err))
            has_bad_names = True
    if has_bad_names:
        sys.exit(2)