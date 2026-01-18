import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _namedtuple_deserialize(dumpable_context: DumpableContext) -> Context:
    class_name = dumpable_context['class_name']
    assert isinstance(class_name, str)
    context = namedtuple(class_name, dumpable_context['fields'])
    return context