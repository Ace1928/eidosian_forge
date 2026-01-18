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
def _generate_altered_foo_together(self, operation):
    for old_value, new_value, app_label, model_name, dependencies in self._get_altered_foo_together_operations(operation.option_name):
        removal_value = new_value.intersection(old_value)
        if new_value != removal_value:
            self.add_operation(app_label, operation(name=model_name, **{operation.option_name: new_value}), dependencies=dependencies)