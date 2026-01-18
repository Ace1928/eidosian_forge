from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
@classmethod
def __get_incoming_edge_map(cls, stream_map):
    incoming_edge_map = {}
    for downstream_label, upstream in list(stream_map.items()):
        incoming_edge_map[downstream_label] = (upstream.node, upstream.label, upstream.selector)
    return incoming_edge_map