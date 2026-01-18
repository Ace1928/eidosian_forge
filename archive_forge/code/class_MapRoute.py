import fnmatch
import re
from collections import OrderedDict
from collections.abc import Mapping
from kombu import Queue
from celery.exceptions import QueueNotFound
from celery.utils.collections import lpmerge
from celery.utils.functional import maybe_evaluate, mlazy
from celery.utils.imports import symbol_by_name
class MapRoute:
    """Creates a router out of a :class:`dict`."""

    def __init__(self, map):
        map = map.items() if isinstance(map, Mapping) else map
        self.map = {}
        self.patterns = OrderedDict()
        for k, v in map:
            if isinstance(k, Pattern):
                self.patterns[k] = v
            elif '*' in k:
                self.patterns[re.compile(fnmatch.translate(k))] = v
            else:
                self.map[k] = v

    def __call__(self, name, *args, **kwargs):
        try:
            return dict(self.map[name])
        except KeyError:
            pass
        except ValueError:
            return {'queue': self.map[name]}
        for regex, route in self.patterns.items():
            if regex.match(name):
                try:
                    return dict(route)
                except ValueError:
                    return {'queue': route}