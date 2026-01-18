import importlib
import os
import re
import sys
from datetime import datetime, timezone
from kombu.utils import json
from kombu.utils.objects import cached_property
from celery import signals
from celery.exceptions import reraise
from celery.utils.collections import DictAttribute, force_mapping
from celery.utils.functional import maybe_list
from celery.utils.imports import NotAPackage, find_module, import_from_cwd, symbol_by_name
def _import_config_module(self, name):
    try:
        self.find_module(name)
    except NotAPackage as exc:
        if name.endswith('.py'):
            reraise(NotAPackage, NotAPackage(CONFIG_WITH_SUFFIX.format(module=name, suggest=name[:-3])), sys.exc_info()[2])
        raise NotAPackage(CONFIG_INVALID_NAME.format(module=name)) from exc
    else:
        return self.import_from_cwd(name)