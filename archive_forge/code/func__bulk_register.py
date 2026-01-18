import abc
from oslo_utils import excutils
from taskflow import logging
from taskflow import states
from taskflow.types import failure
from taskflow.types import notifier
def _bulk_register(watch_states, notifier, cb, details_filter=None):
    """Bulk registers a callback associated with many states."""
    registered = []
    try:
        for state in watch_states:
            if not notifier.is_registered(state, cb, details_filter=details_filter):
                notifier.register(state, cb, details_filter=details_filter)
                registered.append((state, cb))
    except ValueError:
        with excutils.save_and_reraise_exception():
            _bulk_deregister(notifier, registered, details_filter=details_filter)
    else:
        return registered