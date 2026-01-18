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
def _get_app_label_and_model_name(model, app_label=''):
    if isinstance(model, str):
        split = model.split('.', 1)
        return tuple(split) if len(split) == 2 else (app_label, split[0])
    else:
        return (model._meta.app_label, model._meta.model_name)