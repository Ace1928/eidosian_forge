import errno
import os
import shlex
import signal
import sys
from collections import OrderedDict, UserList, defaultdict
from functools import partial
from subprocess import Popen
from time import sleep
from kombu.utils.encoding import from_utf8
from kombu.utils.objects import cached_property
from celery.platforms import IS_WINDOWS, Pidfile, signal_name
from celery.utils.nodenames import gethostname, host_format, node_format, nodesplit
from celery.utils.saferepr import saferepr
class MultiParser:
    Node = Node

    def __init__(self, cmd='celery worker', append='', prefix='', suffix='', range_prefix='celery'):
        self.cmd = cmd
        self.append = append
        self.prefix = prefix
        self.suffix = suffix
        self.range_prefix = range_prefix

    def parse(self, p):
        names = p.values
        options = dict(p.options)
        ranges = len(names) == 1
        prefix = self.prefix
        cmd = options.pop('--cmd', self.cmd)
        append = options.pop('--append', self.append)
        hostname = options.pop('--hostname', options.pop('-n', gethostname()))
        prefix = options.pop('--prefix', prefix) or ''
        suffix = options.pop('--suffix', self.suffix) or hostname
        suffix = '' if suffix in ('""', "''") else suffix
        range_prefix = options.pop('--range-prefix', '') or self.range_prefix
        if ranges:
            try:
                names, prefix = (self._get_ranges(names), range_prefix)
            except ValueError:
                pass
        self._update_ns_opts(p, names)
        self._update_ns_ranges(p, ranges)
        return (self._node_from_options(p, name, prefix, suffix, cmd, append, options) for name in names)

    def _node_from_options(self, p, name, prefix, suffix, cmd, append, options):
        namespace, nodename, _ = build_nodename(name, prefix, suffix)
        namespace = nodename if nodename in p.namespaces else namespace
        return Node(nodename, cmd, append, p.optmerge(namespace, options), p.passthrough)

    def _get_ranges(self, names):
        noderange = int(names[0])
        return [str(n) for n in range(1, noderange + 1)]

    def _update_ns_opts(self, p, names):
        for ns_name, ns_opts in list(p.namespaces.items()):
            if ns_name.isdigit():
                ns_index = int(ns_name) - 1
                if ns_index < 0:
                    raise KeyError(f'Indexes start at 1 got: {ns_name!r}')
                try:
                    p.namespaces[names[ns_index]].update(ns_opts)
                except IndexError:
                    raise KeyError(f'No node at index {ns_name!r}')

    def _update_ns_ranges(self, p, ranges):
        for ns_name, ns_opts in list(p.namespaces.items()):
            if ',' in ns_name or (ranges and '-' in ns_name):
                for subns in self._parse_ns_range(ns_name, ranges):
                    p.namespaces[subns].update(ns_opts)
                p.namespaces.pop(ns_name)

    def _parse_ns_range(self, ns, ranges=False):
        ret = []
        for space in ',' in ns and ns.split(',') or [ns]:
            if ranges and '-' in space:
                start, stop = space.split('-')
                ret.extend((str(n) for n in range(int(start), int(stop) + 1)))
            else:
                ret.append(space)
        return ret