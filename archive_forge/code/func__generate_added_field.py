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
def _generate_added_field(self, app_label, model_name, field_name):
    field = self.to_state.models[app_label, model_name].get_field(field_name)
    dependencies = [(app_label, model_name, field_name, False)]
    if field.remote_field and field.remote_field.model:
        dependencies.extend(self._get_dependencies_for_foreign_key(app_label, model_name, field, self.to_state))
    time_fields = (models.DateField, models.DateTimeField, models.TimeField)
    preserve_default = field.null or field.has_default() or field.db_default is not models.NOT_PROVIDED or field.many_to_many or (field.blank and field.empty_strings_allowed) or (isinstance(field, time_fields) and field.auto_now)
    if not preserve_default:
        field = field.clone()
        if isinstance(field, time_fields) and field.auto_now_add:
            field.default = self.questioner.ask_auto_now_add_addition(field_name, model_name)
        else:
            field.default = self.questioner.ask_not_null_addition(field_name, model_name)
    if field.unique and field.default is not models.NOT_PROVIDED and callable(field.default):
        self.questioner.ask_unique_callable_default_addition(field_name, model_name)
    self.add_operation(app_label, operations.AddField(model_name=model_name, name=field_name, field=field, preserve_default=preserve_default), dependencies=dependencies)