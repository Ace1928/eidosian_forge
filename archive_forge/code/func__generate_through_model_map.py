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
def _generate_through_model_map(self):
    """Through model map generation."""
    for app_label, model_name in sorted(self.old_model_keys):
        old_model_name = self.renamed_models.get((app_label, model_name), model_name)
        old_model_state = self.from_state.models[app_label, old_model_name]
        for field_name, field in old_model_state.fields.items():
            if hasattr(field, 'remote_field') and getattr(field.remote_field, 'through', None):
                through_key = resolve_relation(field.remote_field.through, app_label, model_name)
                self.through_users[through_key] = (app_label, old_model_name, field_name)