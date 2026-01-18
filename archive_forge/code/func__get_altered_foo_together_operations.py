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
def _get_altered_foo_together_operations(self, option_name):
    for app_label, model_name in sorted(self.kept_model_keys):
        old_model_name = self.renamed_models.get((app_label, model_name), model_name)
        old_model_state = self.from_state.models[app_label, old_model_name]
        new_model_state = self.to_state.models[app_label, model_name]
        old_value = old_model_state.options.get(option_name)
        old_value = {tuple((self.renamed_fields.get((app_label, model_name, n), n) for n in unique)) for unique in old_value} if old_value else set()
        new_value = new_model_state.options.get(option_name)
        new_value = set(new_value) if new_value else set()
        if old_value != new_value:
            dependencies = []
            for foo_togethers in new_value:
                for field_name in foo_togethers:
                    field = new_model_state.get_field(field_name)
                    if field.remote_field and field.remote_field.model:
                        dependencies.extend(self._get_dependencies_for_foreign_key(app_label, model_name, field, self.to_state))
            yield (old_value, new_value, app_label, model_name, dependencies)