from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
def _load_redirect(redirect_file):
    try:
        with open(redirect_file) as f:
            yaml = YAML()
            d = yaml.load(f)
    except OSError:
        return {}
    redirect_dict = defaultdict(dict)
    for old_path, new_path in d.items():
        old_class = old_path.split('.')[-1]
        old_module = '.'.join(old_path.split('.')[:-1])
        new_class = new_path.split('.')[-1]
        new_module = '.'.join(new_path.split('.')[:-1])
        redirect_dict[old_module][old_class] = {'@module': new_module, '@class': new_class}
    return dict(redirect_dict)