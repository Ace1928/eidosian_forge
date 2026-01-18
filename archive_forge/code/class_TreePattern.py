from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
class TreePattern(CommonTree):
    """
    When using %label:TOKENNAME in a tree for parse(), we must
    track the label.
    """

    def __init__(self, payload):
        CommonTree.__init__(self, payload)
        self.label = None
        self.hasTextArg = None

    def toString(self):
        if self.label is not None:
            return '%' + self.label + ':' + CommonTree.toString(self)
        else:
            return CommonTree.toString(self)