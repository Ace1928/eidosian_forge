import weakref
from contextlib import contextmanager
from copy import deepcopy
from kombu.utils.imports import symbol_by_name
from celery import Celery, _state
class UnitLogging(symbol_by_name(Celery.log_cls)):
    """Sets up logging for the test application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.already_setup = True