import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def _create_gens(self):
    """Create the generation hashes for route lookups"""
    controllerlist = {}
    actionlist = {}
    for route in self.matchlist:
        if route.static:
            continue
        if 'controller' in route.defaults:
            controllerlist[route.defaults['controller']] = True
        if 'action' in route.defaults:
            actionlist[route.defaults['action']] = True
    controllerlist = list(controllerlist.keys()) + ['*']
    actionlist = list(actionlist.keys()) + ['*']
    gendict = {}
    for route in self.matchlist:
        if route.static:
            continue
        clist = controllerlist
        alist = actionlist
        if 'controller' in route.hardcoded:
            clist = [route.defaults['controller']]
        if 'action' in route.hardcoded:
            alist = [six.text_type(route.defaults['action'])]
        for controller in clist:
            for action in alist:
                actiondict = gendict.setdefault(controller, {})
                actiondict.setdefault(action, ([], {}))[0].append(route)
    self._gendict = gendict
    self._created_gens = True