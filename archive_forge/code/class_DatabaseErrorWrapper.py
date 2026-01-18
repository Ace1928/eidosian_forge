import pkgutil
from importlib import import_module
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.connection import ConnectionDoesNotExist  # NOQA: F401
from django.utils.connection import BaseConnectionHandler
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class DatabaseErrorWrapper:
    """
    Context manager and decorator that reraises backend-specific database
    exceptions using Django's common wrappers.
    """

    def __init__(self, wrapper):
        """
        wrapper is a database wrapper.

        It must have a Database attribute defining PEP-249 exceptions.
        """
        self.wrapper = wrapper

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return
        for dj_exc_type in (DataError, OperationalError, IntegrityError, InternalError, ProgrammingError, NotSupportedError, DatabaseError, InterfaceError, Error):
            db_exc_type = getattr(self.wrapper.Database, dj_exc_type.__name__)
            if issubclass(exc_type, db_exc_type):
                dj_exc_value = dj_exc_type(*exc_value.args)
                if dj_exc_type not in (DataError, IntegrityError):
                    self.wrapper.errors_occurred = True
                raise dj_exc_value.with_traceback(traceback) from exc_value

    def __call__(self, func):

        def inner(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return inner