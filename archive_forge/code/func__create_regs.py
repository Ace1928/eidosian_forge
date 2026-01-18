import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def _create_regs(self, clist=None):
    """Creates regular expressions for all connected routes"""
    if clist is None:
        if self.directory:
            clist = self.controller_scan(self.directory)
        elif callable(self.controller_scan):
            clist = self.controller_scan()
        elif not self.controller_scan:
            clist = []
        else:
            clist = self.controller_scan
    for key, val in six.iteritems(self.maxkeys):
        for route in val:
            route.makeregexp(clist)
    regexps = []
    prefix2routes = collections.defaultdict(list)
    for route in self.matchlist:
        if not route.static:
            regexps.append(route.makeregexp(clist, include_names=False))
            prefix = ''.join(it.takewhile(lambda p: isinstance(p, str), route.routelist))
            if route.minimization and (not prefix.startswith('/')):
                prefix = '/' + prefix
            prefix2routes[prefix.rstrip('/')].append(route)
    self._prefix2routes = prefix2routes
    self._prefix_lens = sorted(set((len(p) for p in prefix2routes)), reverse=True)
    if self.prefix:
        self._regprefix = re.compile(self.prefix + '(.*)')
    regexp = '|'.join(['(?:%s)' % x for x in regexps])
    self._master_reg = regexp
    try:
        self._master_regexp = re.compile(regexp)
    except OverflowError:
        self._master_regexp = None
    self._created_regs = True