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
def _check_supported(self, databases):
    errors = []
    for db in databases:
        if not router.allow_migrate_model(db, self.model):
            continue
        connection = connections[db]
        if self.model._meta.required_db_vendor and self.model._meta.required_db_vendor != connection.vendor:
            continue
        if not ('supports_json_field' in self.model._meta.required_db_features or connection.features.supports_json_field):
            errors.append(checks.Error('%s does not support JSONFields.' % connection.display_name, obj=self.model, id='fields.E180'))
    return errors