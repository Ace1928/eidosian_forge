import weakref
from contextlib import contextmanager
from copy import deepcopy
from kombu.utils.imports import symbol_by_name
from celery import Celery, _state
class NonTLS:
    current_app = trap