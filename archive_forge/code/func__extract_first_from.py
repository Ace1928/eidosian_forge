import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
def _extract_first_from(name, sources):
    """Extracts/returns first occurrence of key in list of dicts."""
    for i, source in enumerate(sources):
        if not source:
            continue
        if name in source:
            return (i, source[name])
    raise KeyError(name)