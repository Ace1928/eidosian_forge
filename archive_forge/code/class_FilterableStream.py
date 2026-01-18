from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
class FilterableStream(Stream):

    def __init__(self, upstream_node, upstream_label, upstream_selector=None):
        super(FilterableStream, self).__init__(upstream_node, upstream_label, {InputNode, FilterNode}, upstream_selector)