import sys
import warnings
import importlib
from contextlib import contextmanager
import gi
from ._gi import Repository, RepositoryError
from ._gi import PyGIWarning
from .module import get_introspection_module
from .overrides import load_overrides
@contextmanager
def _check_require_version(namespace, stacklevel):
    """A context manager which tries to give helpful warnings
    about missing gi.require_version() which could potentially
    break code if only an older version than expected is installed
    or a new version gets introduced.

    ::

        with _check_require_version("Gtk", stacklevel):
            load_namespace_and_overrides()
    """
    was_loaded = repository.is_registered(namespace)
    yield
    if was_loaded:
        return
    if namespace in ('GLib', 'GObject', 'Gio'):
        return
    if gi.get_required_version(namespace) is not None:
        return
    version = repository.get_version(namespace)
    warnings.warn("%(namespace)s was imported without specifying a version first. Use gi.require_version('%(namespace)s', '%(version)s') before import to ensure that the right version gets loaded." % {'namespace': namespace, 'version': version}, PyGIWarning, stacklevel=stacklevel)