import itertools
import time
from oslo_utils import timeutils
from taskflow.engines.action_engine import compiler as co
from taskflow import exceptions as exc
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
def _record_atom_event(self, state, atom_name):
    meta_update = {'%s-timestamp' % state: time.time()}
    try:
        self._engine.storage.update_atom_metadata(atom_name, meta_update)
    except exc.StorageFailure:
        LOG.warning('Failure to store timestamp %s for atom %s', meta_update, atom_name, exc_info=True)