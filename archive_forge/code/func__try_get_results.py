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
def _try_get_results(self, looking_for, provider, look_into_results=True, find_potentials=False):
    if provider.name is _TRANSIENT_PROVIDER:
        results = self.transient_results
    else:
        try:
            results = self.result_fetcher(provider.name)
        except (exceptions.NotFound, exceptions.DisallowedAccess):
            if not find_potentials:
                raise
            else:
                results = {}
    if look_into_results:
        _item_from_single(provider, results, looking_for)
    return results