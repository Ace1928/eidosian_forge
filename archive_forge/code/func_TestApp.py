import weakref
from contextlib import contextmanager
from copy import deepcopy
from kombu.utils.imports import symbol_by_name
from celery import Celery, _state
def TestApp(name=None, config=None, enable_logging=False, set_as_current=False, log=UnitLogging, backend=None, broker=None, **kwargs):
    """App used for testing."""
    from . import tasks
    config = dict(deepcopy(DEFAULT_TEST_CONFIG), **config or {})
    if broker is not None:
        config.pop('broker_url', None)
    if backend is not None:
        config.pop('result_backend', None)
    log = None if enable_logging else log
    test_app = Celery(name or 'celery.tests', set_as_current=set_as_current, log=log, broker=broker, backend=backend, **kwargs)
    test_app.add_defaults(config)
    return test_app