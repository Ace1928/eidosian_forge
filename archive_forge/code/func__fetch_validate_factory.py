import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def _fetch_validate_factory(flow_factory):
    if isinstance(flow_factory, str):
        factory_fun = _fetch_factory(flow_factory)
        factory_name = flow_factory
    else:
        factory_fun = flow_factory
        factory_name = reflection.get_callable_name(flow_factory)
        try:
            reimported = _fetch_factory(factory_name)
            assert reimported == factory_fun
        except (ImportError, AssertionError):
            raise ValueError('Flow factory %r is not reimportable by name %s' % (factory_fun, factory_name))
    return (factory_name, factory_fun)