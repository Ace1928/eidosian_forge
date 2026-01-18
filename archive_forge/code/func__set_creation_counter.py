import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
def _set_creation_counter(self):
    """
        Set the creation counter value for this instance and increment the
        class-level copy.
        """
    self.creation_counter = BaseManager.creation_counter
    BaseManager.creation_counter += 1