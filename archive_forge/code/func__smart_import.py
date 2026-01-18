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
def _smart_import(self, path, imp=None):
    imp = self.import_module if imp is None else imp
    if ':' in path:
        return symbol_by_name(path, imp=imp)
    try:
        return imp(path)
    except ImportError:
        return symbol_by_name(path, imp=imp)