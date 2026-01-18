import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
def _moved_msg(new_name: str, old_name: ty.Optional[str]) -> None:
    if old_name:
        deprecated_msg = "Property '%(old_name)s' has moved to '%(new_name)s'"
        deprecated_msg = deprecated_msg % {'old_name': old_name, 'new_name': new_name}
        debtcollector.deprecate(deprecated_msg, version='2.6', removal_version='3.0', stacklevel=5)