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
@staticmethod
def _resolve_dependency(dependency):
    """
        Return the resolved dependency and a boolean denoting whether or not
        it was swappable.
        """
    if dependency[0] != '__setting__':
        return (dependency, False)
    resolved_app_label, resolved_object_name = getattr(settings, dependency[1]).split('.')
    return ((resolved_app_label, resolved_object_name.lower()) + dependency[2:], True)