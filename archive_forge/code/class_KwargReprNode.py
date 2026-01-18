from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
class KwargReprNode(DagNode):
    """A DagNode that can be represented as a set of args+kwargs.
    """

    @property
    def __upstream_hashes(self):
        hashes = []
        for downstream_label, upstream_info in list(self.incoming_edge_map.items()):
            upstream_node, upstream_label, upstream_selector = upstream_info
            hashes += [hash(x) for x in [downstream_label, upstream_node, upstream_label, upstream_selector]]
        return hashes

    @property
    def __inner_hash(self):
        props = {'args': self.args, 'kwargs': self.kwargs}
        return get_hash(props)

    def __get_hash(self):
        hashes = self.__upstream_hashes + [self.__inner_hash]
        return get_hash_int(hashes)

    def __init__(self, incoming_edge_map, name, args, kwargs):
        self.__incoming_edge_map = incoming_edge_map
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.__hash = self.__get_hash()

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def short_hash(self):
        return '{:x}'.format(abs(hash(self)))[:12]

    def long_repr(self, include_hash=True):
        formatted_props = ['{!r}'.format(arg) for arg in self.args]
        formatted_props += ['{}={!r}'.format(key, self.kwargs[key]) for key in sorted(self.kwargs)]
        out = '{}({})'.format(self.name, ', '.join(formatted_props))
        if include_hash:
            out += ' <{}>'.format(self.short_hash)
        return out

    def __repr__(self):
        return self.long_repr()

    @property
    def incoming_edges(self):
        return get_incoming_edges(self, self.incoming_edge_map)

    @property
    def incoming_edge_map(self):
        return self.__incoming_edge_map

    @property
    def short_repr(self):
        return self.name