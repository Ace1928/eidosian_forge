from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class StandardParser(object):

    def __init__(self, npargs=None, kwargs=None, flags=None, doc=None):
        if npargs is None:
            npargs = '*'
        if flags is None:
            flags = []
        if kwargs is None:
            kwargs = {}
        self.npargs = npargs
        self.kwargs = kwargs
        self.flags = flags
        self.doc = doc

    def __call__(self, ctx, tokens, breakstack):
        return StandardArgTree.parse(ctx, tokens, self.npargs, self.kwargs, self.flags, breakstack)