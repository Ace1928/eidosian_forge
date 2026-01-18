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
@staticmethod
def _create_atom_detail(atom_name, atom_detail_cls, atom_version=None, atom_state=states.PENDING):
    ad = atom_detail_cls(atom_name, uuidutils.generate_uuid())
    ad.state = atom_state
    if atom_version is not None:
        ad.version = atom_version
    return ad