import datetime
import importlib
import os
import sys
from django.apps import apps
from django.core.management.base import OutputWrapper
from django.db.models import NOT_PROVIDED
from django.utils import timezone
from django.utils.version import get_docs_version
from .loader import MigrationLoader
class NonInteractiveMigrationQuestioner(MigrationQuestioner):

    def __init__(self, defaults=None, specified_apps=None, dry_run=None, verbosity=1, log=None):
        self.verbosity = verbosity
        self.log = log
        super().__init__(defaults=defaults, specified_apps=specified_apps, dry_run=dry_run)

    def log_lack_of_migration(self, field_name, model_name, reason):
        if self.verbosity > 0:
            self.log(f"Field '{field_name}' on model '{model_name}' not migrated: {reason}.")

    def ask_not_null_addition(self, field_name, model_name):
        self.log_lack_of_migration(field_name, model_name, 'it is impossible to add a non-nullable field without specifying a default')
        sys.exit(3)

    def ask_not_null_alteration(self, field_name, model_name):
        self.log(f"Field '{field_name}' on model '{model_name}' given a default of NOT PROVIDED and must be corrected.")
        return NOT_PROVIDED

    def ask_auto_now_add_addition(self, field_name, model_name):
        self.log_lack_of_migration(field_name, model_name, "it is impossible to add a field with 'auto_now_add=True' without specifying a default")
        sys.exit(3)