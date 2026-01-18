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
def _detect_changes(self, convert_apps=None, graph=None):
    """
        Return a dict of migration plans which will achieve the
        change from from_state to to_state. The dict has app labels
        as keys and a list of migrations as values.

        The resulting migrations aren't specially named, but the names
        do matter for dependencies inside the set.

        convert_apps is the list of apps to convert to use migrations
        (i.e. to make initial migrations for, in the usual case)

        graph is an optional argument that, if provided, can help improve
        dependency generation and avoid potential circular dependencies.
        """
    self.generated_operations = {}
    self.altered_indexes = {}
    self.altered_constraints = {}
    self.renamed_fields = {}
    self.old_model_keys = set()
    self.old_proxy_keys = set()
    self.old_unmanaged_keys = set()
    self.new_model_keys = set()
    self.new_proxy_keys = set()
    self.new_unmanaged_keys = set()
    for (app_label, model_name), model_state in self.from_state.models.items():
        if not model_state.options.get('managed', True):
            self.old_unmanaged_keys.add((app_label, model_name))
        elif app_label not in self.from_state.real_apps:
            if model_state.options.get('proxy'):
                self.old_proxy_keys.add((app_label, model_name))
            else:
                self.old_model_keys.add((app_label, model_name))
    for (app_label, model_name), model_state in self.to_state.models.items():
        if not model_state.options.get('managed', True):
            self.new_unmanaged_keys.add((app_label, model_name))
        elif app_label not in self.from_state.real_apps or (convert_apps and app_label in convert_apps):
            if model_state.options.get('proxy'):
                self.new_proxy_keys.add((app_label, model_name))
            else:
                self.new_model_keys.add((app_label, model_name))
    self.from_state.resolve_fields_and_relations()
    self.to_state.resolve_fields_and_relations()
    self.generate_renamed_models()
    self._prepare_field_lists()
    self._generate_through_model_map()
    self.generate_deleted_models()
    self.generate_created_models()
    self.generate_deleted_proxies()
    self.generate_created_proxies()
    self.generate_altered_options()
    self.generate_altered_managers()
    self.generate_altered_db_table_comment()
    self.create_renamed_fields()
    self.create_altered_indexes()
    self.create_altered_constraints()
    self.generate_removed_constraints()
    self.generate_removed_indexes()
    self.generate_renamed_fields()
    self.generate_renamed_indexes()
    self.generate_removed_altered_unique_together()
    self.generate_removed_altered_index_together()
    self.generate_removed_fields()
    self.generate_added_fields()
    self.generate_altered_fields()
    self.generate_altered_order_with_respect_to()
    self.generate_altered_unique_together()
    self.generate_altered_index_together()
    self.generate_added_indexes()
    self.generate_added_constraints()
    self.generate_altered_db_table()
    self._sort_migrations()
    self._build_migration_list(graph)
    self._optimize_migrations()
    return self.migrations