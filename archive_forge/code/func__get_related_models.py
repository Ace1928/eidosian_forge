import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from django.apps import AppConfig
from django.apps.registry import Apps
from django.apps.registry import apps as global_apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.migrations.utils import field_is_referenced, get_references
from django.db.models import NOT_PROVIDED
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
from django.db.models.options import DEFAULT_NAMES, normalize_together
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.version import get_docs_version
from .exceptions import InvalidBasesError
from .utils import resolve_relation
def _get_related_models(m):
    """Return all models that have a direct relationship to the given model."""
    related_models = [subclass for subclass in m.__subclasses__() if issubclass(subclass, models.Model)]
    related_fields_models = set()
    for f in m._meta.get_fields(include_parents=True, include_hidden=True):
        if f.is_relation and f.related_model is not None and (not isinstance(f.related_model, str)):
            related_fields_models.add(f.model)
            related_models.append(f.related_model)
    opts = m._meta
    if opts.proxy and m in related_fields_models:
        related_models.append(opts.concrete_model)
    return related_models