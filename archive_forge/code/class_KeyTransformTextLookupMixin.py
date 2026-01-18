import json
import warnings
from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import expressions, lookups
from django.db.models.constants import LOOKUP_SEP
from django.db.models.fields import TextField
from django.db.models.lookups import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import CheckFieldDefaultMixin
class KeyTransformTextLookupMixin:
    """
    Mixin for combining with a lookup expecting a text lhs from a JSONField
    key lookup. On PostgreSQL, make use of the ->> operator instead of casting
    key values to text and performing the lookup on the resulting
    representation.
    """

    def __init__(self, key_transform, *args, **kwargs):
        if not isinstance(key_transform, KeyTransform):
            raise TypeError('Transform should be an instance of KeyTransform in order to use this lookup.')
        key_text_transform = KeyTextTransform(key_transform.key_name, *key_transform.source_expressions, **key_transform.extra)
        super().__init__(key_text_transform, *args, **kwargs)