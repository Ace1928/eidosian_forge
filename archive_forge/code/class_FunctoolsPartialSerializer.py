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
class FunctoolsPartialSerializer(BaseSerializer):

    def serialize(self):
        func_string, func_imports = serializer_factory(self.value.func).serialize()
        args_string, args_imports = serializer_factory(self.value.args).serialize()
        keywords_string, keywords_imports = serializer_factory(self.value.keywords).serialize()
        imports = {'import functools', *func_imports, *args_imports, *keywords_imports}
        return ('functools.%s(%s, *%s, **%s)' % (self.value.__class__.__name__, func_string, args_string, keywords_string), imports)