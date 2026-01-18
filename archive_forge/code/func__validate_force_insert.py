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
def _validate_force_insert(cls, force_insert):
    if force_insert is False:
        return ()
    if force_insert is True:
        return (cls,)
    if not isinstance(force_insert, tuple):
        raise TypeError('force_insert must be a bool or tuple.')
    for member in force_insert:
        if not isinstance(member, ModelBase):
            raise TypeError(f'Invalid force_insert member. {member!r} must be a model subclass.')
        if not issubclass(cls, member):
            raise TypeError(f'Invalid force_insert member. {member.__qualname__} must be a base of {cls.__qualname__}.')
    return force_insert