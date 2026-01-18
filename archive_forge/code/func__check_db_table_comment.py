import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
from asgiref.sync import sync_to_async
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
from django.db import (
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import DatabaseDefault, RawSQL
from django.db.models.fields.related import (
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
@classmethod
def _check_db_table_comment(cls, databases):
    if not cls._meta.db_table_comment:
        return []
    errors = []
    for db in databases:
        if not router.allow_migrate_model(db, cls):
            continue
        connection = connections[db]
        if not (connection.features.supports_comments or 'supports_comments' in cls._meta.required_db_features):
            errors.append(checks.Warning(f'{connection.display_name} does not support comments on tables (db_table_comment).', obj=cls, id='models.W046'))
    return errors