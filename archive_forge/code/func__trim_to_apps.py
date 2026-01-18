import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def _trim_to_apps(self, changes, app_labels):
    """
        Take changes from arrange_for_graph() and set of app labels, and return
        a modified set of changes which trims out as many migrations that are
        not in app_labels as possible. Note that some other migrations may
        still be present as they may be required dependencies.
        """
    app_dependencies = {}
    for app_label, migrations in changes.items():
        for migration in migrations:
            for dep_app_label, name in migration.dependencies:
                app_dependencies.setdefault(app_label, set()).add(dep_app_label)
    required_apps = set(app_labels)
    old_required_apps = None
    while old_required_apps != required_apps:
        old_required_apps = set(required_apps)
        required_apps.update(*[app_dependencies.get(app_label, ()) for app_label in required_apps])
    for app_label in list(changes):
        if app_label not in required_apps:
            del changes[app_label]
    return changes