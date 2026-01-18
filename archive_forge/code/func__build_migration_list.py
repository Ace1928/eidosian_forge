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
def _build_migration_list(self, graph=None):
    """
        Chop the lists of operations up into migrations with dependencies on
        each other. Do this by going through an app's list of operations until
        one is found that has an outgoing dependency that isn't in another
        app's migration yet (hasn't been chopped off its list). Then chop off
        the operations before it into a migration and move onto the next app.
        If the loops completes without doing anything, there's a circular
        dependency (which _should_ be impossible as the operations are
        all split at this point so they can't depend and be depended on).
        """
    self.migrations = {}
    num_ops = sum((len(x) for x in self.generated_operations.values()))
    chop_mode = False
    while num_ops:
        for app_label in sorted(self.generated_operations):
            chopped = []
            dependencies = set()
            for operation in list(self.generated_operations[app_label]):
                deps_satisfied = True
                operation_dependencies = set()
                for dep in operation._auto_deps:
                    original_dep = dep
                    dep, is_swappable_dep = self._resolve_dependency(dep)
                    if dep[0] != app_label:
                        for other_operation in self.generated_operations.get(dep[0], []):
                            if self.check_dependency(other_operation, dep):
                                deps_satisfied = False
                                break
                        if not deps_satisfied:
                            break
                        elif is_swappable_dep:
                            operation_dependencies.add((original_dep[0], original_dep[1]))
                        elif dep[0] in self.migrations:
                            operation_dependencies.add((dep[0], self.migrations[dep[0]][-1].name))
                        elif chop_mode:
                            if graph and graph.leaf_nodes(dep[0]):
                                operation_dependencies.add(graph.leaf_nodes(dep[0])[0])
                            else:
                                operation_dependencies.add((dep[0], '__first__'))
                        else:
                            deps_satisfied = False
                if deps_satisfied:
                    chopped.append(operation)
                    dependencies.update(operation_dependencies)
                    del self.generated_operations[app_label][0]
                else:
                    break
            if dependencies or chopped:
                if not self.generated_operations[app_label] or chop_mode:
                    subclass = type('Migration', (Migration,), {'operations': [], 'dependencies': []})
                    instance = subclass('auto_%i' % (len(self.migrations.get(app_label, [])) + 1), app_label)
                    instance.dependencies = list(dependencies)
                    instance.operations = chopped
                    instance.initial = app_label not in self.existing_apps
                    self.migrations.setdefault(app_label, []).append(instance)
                    chop_mode = False
                else:
                    self.generated_operations[app_label] = chopped + self.generated_operations[app_label]
        new_num_ops = sum((len(x) for x in self.generated_operations.values()))
        if new_num_ops == num_ops:
            if not chop_mode:
                chop_mode = True
            else:
                raise ValueError('Cannot resolve operation dependencies: %r' % self.generated_operations)
        num_ops = new_num_ops