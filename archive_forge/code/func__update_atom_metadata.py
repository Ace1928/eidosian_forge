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
@fasteners.write_locked
def _update_atom_metadata(self, atom_name, update_with, expected_type=None):
    source, clone = self._atomdetail_by_name(atom_name, expected_type=expected_type, clone=True)
    if update_with:
        clone.meta.update(update_with)
        self._with_connection(self._save_atom_detail, source, clone)