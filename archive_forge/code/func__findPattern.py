from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
def _findPattern(self, t, pattern):
    """Return a List of subtrees matching pattern."""
    subtrees = []
    tokenizer = TreePatternLexer(pattern)
    parser = TreePatternParser(tokenizer, self, TreePatternTreeAdaptor())
    tpattern = parser.pattern()
    if tpattern is None or tpattern.isNil() or isinstance(tpattern, WildcardTreePattern):
        return None
    rootTokenType = tpattern.getType()

    def visitor(tree, parent, childIndex, label):
        if self._parse(tree, tpattern, None):
            subtrees.append(tree)
    self.visit(t, rootTokenType, visitor)
    return subtrees