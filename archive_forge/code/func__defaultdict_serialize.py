import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _defaultdict_serialize(context: Context) -> DumpableContext:
    default_factory, dict_context = context
    json_defaultdict = {'default_factory_module': default_factory.__module__, 'default_factory_name': default_factory.__qualname__, 'dict_context': dict_context}
    return json_defaultdict