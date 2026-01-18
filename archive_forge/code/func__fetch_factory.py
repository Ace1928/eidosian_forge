import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def _fetch_factory(factory_name):
    try:
        return importutils.import_class(factory_name)
    except (ImportError, ValueError) as e:
        raise ImportError('Could not import factory %r: %s' % (factory_name, e))