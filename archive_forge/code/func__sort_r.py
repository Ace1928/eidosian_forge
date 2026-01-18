from suds import *
from logging import getLogger
def _sort_r(sorted, processed, key, deps, dependency_tree):
    """Recursive topological sort implementation."""
    if key in processed:
        return
    processed.add(key)
    for dep_key in deps:
        dep_deps = dependency_tree.get(dep_key)
        if dep_deps is None:
            log.debug('"%s" not found, skipped', Repr(dep_key))
            continue
        _sort_r(sorted, processed, dep_key, dep_deps, dependency_tree)
    sorted.append((key, deps))