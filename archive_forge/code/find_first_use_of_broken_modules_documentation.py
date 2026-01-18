from typing import Dict, List
from ..package_exporter import PackagingError

    Find all broken modules in a PackagingError, and for each one, return the
    dependency path in which the module was first encountered.

    E.g. broken module m.n.o was added to a dependency graph while processing a.b.c,
    then re-encountered while processing d.e.f. This method would return
    {'m.n.o': ['a', 'b', 'c']}

    Args:
        exc: a PackagingError

    Returns: A dict from broken module names to lists of module names in the path.
    