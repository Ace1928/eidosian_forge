from __future__ import annotations
import sys
import eventlet
def _upgrade_instances(container, klass, upgrade, visited=None, old_to_new=None):
    """
    Starting with a Python object, find all instances of ``klass``, following
    references in ``dict`` values, ``list`` items, and attributes.

    Once an object is found, replace all instances with
    ``upgrade(found_object)``, again limited to the criteria above.

    In practice this is used only for ``threading.RLock``, so we can assume
    instances are hashable.
    """
    if visited is None:
        visited = {}
    if old_to_new is None:
        old_to_new = {}
    visited[id(container)] = container

    def upgrade_or_traverse(obj):
        if id(obj) in visited:
            return None
        if isinstance(obj, klass):
            if obj in old_to_new:
                return old_to_new[obj]
            else:
                new = upgrade(obj)
                old_to_new[obj] = new
                return new
        else:
            _upgrade_instances(obj, klass, upgrade, visited, old_to_new)
            return None
    if isinstance(container, dict):
        for k, v in list(container.items()):
            new = upgrade_or_traverse(v)
            if new is not None:
                container[k] = new
    if isinstance(container, list):
        for i, v in enumerate(container):
            new = upgrade_or_traverse(v)
            if new is not None:
                container[i] = new
    try:
        container_vars = vars(container)
    except TypeError:
        pass
    else:
        try:
            for k, v in list(container_vars.items()):
                new = upgrade_or_traverse(v)
                if new is not None:
                    setattr(container, k, new)
        except:
            import logging
            logger = logging.Logger('eventlet')
            logger.exception('An exception was thrown while monkey_patching for eventlet. to fix this error make sure you run eventlet.monkey_patch() before importing any other modules.', exc_info=True)