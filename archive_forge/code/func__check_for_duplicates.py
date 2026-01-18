from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
def _check_for_duplicates(arg_name, objs):
    used_vals = set()
    for val in objs:
        if val in used_vals:
            raise ValueError('Found duplicate value %s in CreateModel %s argument.' % (val, arg_name))
        used_vals.add(val)