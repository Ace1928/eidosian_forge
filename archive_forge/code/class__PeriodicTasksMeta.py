import copy
import logging
import random
import time
from time import monotonic as now
from oslo_service._i18n import _
from oslo_service import _options
from oslo_utils import reflection
class _PeriodicTasksMeta(type):

    def _add_periodic_task(cls, task):
        """Add a periodic task to the list of periodic tasks.

        The task should already be decorated by @periodic_task.

        :return: whether task was actually enabled
        """
        name = task._periodic_name
        if task._periodic_spacing < 0:
            LOG.info('Skipping periodic task %(task)s because its interval is negative', {'task': name})
            return False
        if not task._periodic_enabled:
            LOG.info('Skipping periodic task %(task)s because it is disabled', {'task': name})
            return False
        if task._periodic_spacing == 0:
            task._periodic_spacing = DEFAULT_INTERVAL
        cls._periodic_tasks.append((name, task))
        cls._periodic_spacing[name] = task._periodic_spacing
        return True

    def __init__(cls, names, bases, dict_):
        """Metaclass that allows us to collect decorated periodic tasks."""
        super(_PeriodicTasksMeta, cls).__init__(names, bases, dict_)
        try:
            cls._periodic_tasks = cls._periodic_tasks[:]
        except AttributeError:
            cls._periodic_tasks = []
        try:
            cls._periodic_spacing = cls._periodic_spacing.copy()
        except AttributeError:
            cls._periodic_spacing = {}
        for value in cls.__dict__.values():
            if getattr(value, '_periodic_task', False):
                cls._add_periodic_task(value)