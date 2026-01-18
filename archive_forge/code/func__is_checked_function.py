import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False
    if item.__name__.startswith('_'):
        return False
    mod = item.__module__
    if not mod.startswith('sklearn.') or mod.endswith('estimator_checks'):
        return False
    return True