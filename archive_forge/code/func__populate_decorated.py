import warnings
from warnings import warn
import breezy
def _populate_decorated(callable, deprecation_version, label, decorated_callable):
    """Populate attributes like __name__ and __doc__ on the decorated callable.
    """
    _decorate_docstring(callable, deprecation_version, label, decorated_callable)
    decorated_callable.__module__ = callable.__module__
    decorated_callable.__name__ = callable.__name__
    decorated_callable.is_deprecated = True