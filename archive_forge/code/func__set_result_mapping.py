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
def _set_result_mapping(self, provider_name, mapping):
    """Sets the result mapping for a given producer.

        The result saved with given name would be accessible by names
        defined in mapping. Mapping is a dict name => index. If index
        is None, the whole result will have this name; else, only
        part of it, result[index].
        """
    provider_mapping = self._result_mappings.setdefault(provider_name, {})
    if mapping:
        provider_mapping.update(mapping)
        for name, index in provider_mapping.items():
            entries = self._reverse_mapping.setdefault(name, [])
            provider = _Provider(provider_name, index)
            if provider not in entries:
                entries.append(provider)