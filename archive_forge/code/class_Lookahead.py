from __future__ import unicode_literals
import re
class Lookahead(Node):
    """
    Lookahead expression.
    """

    def __init__(self, childnode, negative=False):
        self.childnode = childnode
        self.negative = negative

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.childnode)