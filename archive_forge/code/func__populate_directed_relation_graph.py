import bisect
import copy
import inspect
import warnings
from collections import defaultdict
from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import AutoField, Manager, OrderWrt, UniqueConstraint
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override
def _populate_directed_relation_graph(self):
    """
        This method is used by each model to find its reverse objects. As this
        method is very expensive and is accessed frequently (it looks up every
        field in a model, in every app), it is computed on first access and then
        is set as a property on every model.
        """
    related_objects_graph = defaultdict(list)
    all_models = self.apps.get_models(include_auto_created=True)
    for model in all_models:
        opts = model._meta
        if opts.abstract:
            continue
        fields_with_relations = (f for f in opts._get_fields(reverse=False, include_parents=False) if f.is_relation and f.related_model is not None)
        for f in fields_with_relations:
            if not isinstance(f.remote_field.model, str):
                remote_label = f.remote_field.model._meta.concrete_model._meta.label
                related_objects_graph[remote_label].append(f)
    for model in all_models:
        related_objects = related_objects_graph[model._meta.concrete_model._meta.label]
        model._meta.__dict__['_relation_tree'] = related_objects
    return self.__dict__.get('_relation_tree', EMPTY_RELATION_TREE)