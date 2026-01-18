import builtins
import collections.abc
import datetime
import decimal
import enum
import functools
import math
import os
import pathlib
import re
import types
import uuid
from django.conf import SettingsReference
from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.utils import COMPILED_REGEX_TYPE, RegexObject
from django.utils.functional import LazyObject, Promise
from django.utils.version import PY311, get_docs_version
class TupleSerializer(BaseSequenceSerializer):

    def _format(self):
        return '(%s)' if len(self.value) != 1 else '(%s,)'