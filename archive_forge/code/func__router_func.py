import pkgutil
from importlib import import_module
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.connection import ConnectionDoesNotExist  # NOQA: F401
from django.utils.connection import BaseConnectionHandler
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def _router_func(action):

    def _route_db(self, model, **hints):
        chosen_db = None
        for router in self.routers:
            try:
                method = getattr(router, action)
            except AttributeError:
                pass
            else:
                chosen_db = method(model, **hints)
                if chosen_db:
                    return chosen_db
        instance = hints.get('instance')
        if instance is not None and instance._state.db:
            return instance._state.db
        return DEFAULT_DB_ALIAS
    return _route_db