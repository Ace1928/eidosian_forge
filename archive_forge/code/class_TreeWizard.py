from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
class TreeWizard(object):
    """
    Build and navigate trees with this object.  Must know about the names
    of tokens so you have to pass in a map or array of token names (from which
    this class can build the map).  I.e., Token DECL means nothing unless the
    class can translate it to a token type.

    In order to create nodes and navigate, this class needs a TreeAdaptor.

    This class can build a token type -> node index for repeated use or for
    iterating over the various nodes with a particular type.

    This class works in conjunction with the TreeAdaptor rather than moving
    all this functionality into the adaptor.  An adaptor helps build and
    navigate trees using methods.  This class helps you do it with string
    patterns like "(A B C)".  You can create a tree from that pattern or
    match subtrees against it.
    """

    def __init__(self, adaptor=None, tokenNames=None, typeMap=None):
        self.adaptor = adaptor
        if typeMap is None:
            self.tokenNameToTypeMap = computeTokenTypes(tokenNames)
        else:
            if tokenNames is not None:
                raise ValueError("Can't have both tokenNames and typeMap")
            self.tokenNameToTypeMap = typeMap

    def getTokenType(self, tokenName):
        """Using the map of token names to token types, return the type."""
        try:
            return self.tokenNameToTypeMap[tokenName]
        except KeyError:
            return INVALID_TOKEN_TYPE

    def create(self, pattern):
        """
        Create a tree or node from the indicated tree pattern that closely
        follows ANTLR tree grammar tree element syntax:

        (root child1 ... child2).

        You can also just pass in a node: ID

        Any node can have a text argument: ID[foo]
        (notice there are no quotes around foo--it's clear it's a string).

        nil is a special name meaning "give me a nil node".  Useful for
        making lists: (nil A B C) is a list of A B C.
        """
        tokenizer = TreePatternLexer(pattern)
        parser = TreePatternParser(tokenizer, self, self.adaptor)
        return parser.pattern()

    def index(self, tree):
        """Walk the entire tree and make a node name to nodes mapping.

        For now, use recursion but later nonrecursive version may be
        more efficient.  Returns a dict int -> list where the list is
        of your AST node type.  The int is the token type of the node.
        """
        m = {}
        self._index(tree, m)
        return m

    def _index(self, t, m):
        """Do the work for index"""
        if t is None:
            return
        ttype = self.adaptor.getType(t)
        elements = m.get(ttype)
        if elements is None:
            m[ttype] = elements = []
        elements.append(t)
        for i in range(self.adaptor.getChildCount(t)):
            child = self.adaptor.getChild(t, i)
            self._index(child, m)

    def find(self, tree, what):
        """Return a list of matching token.

        what may either be an integer specifzing the token type to find or
        a string with a pattern that must be matched.

        """
        if isinstance(what, six.integer_types):
            return self._findTokenType(tree, what)
        elif isinstance(what, six.string_types):
            return self._findPattern(tree, what)
        else:
            raise TypeError("'what' must be string or integer")

    def _findTokenType(self, t, ttype):
        """Return a List of tree nodes with token type ttype"""
        nodes = []

        def visitor(tree, parent, childIndex, labels):
            nodes.append(tree)
        self.visit(t, ttype, visitor)
        return nodes

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

    def visit(self, tree, what, visitor):
        """Visit every node in tree matching what, invoking the visitor.

        If what is a string, it is parsed as a pattern and only matching
        subtrees will be visited.
        The implementation uses the root node of the pattern in combination
        with visit(t, ttype, visitor) so nil-rooted patterns are not allowed.
        Patterns with wildcard roots are also not allowed.

        If what is an integer, it is used as a token type and visit will match
        all nodes of that type (this is faster than the pattern match).
        The labels arg of the visitor action method is never set (it's None)
        since using a token type rather than a pattern doesn't let us set a
        label.
        """
        if isinstance(what, six.integer_types):
            self._visitType(tree, None, 0, what, visitor)
        elif isinstance(what, six.string_types):
            self._visitPattern(tree, what, visitor)
        else:
            raise TypeError("'what' must be string or integer")

    def _visitType(self, t, parent, childIndex, ttype, visitor):
        """Do the recursive work for visit"""
        if t is None:
            return
        if self.adaptor.getType(t) == ttype:
            visitor(t, parent, childIndex, None)
        for i in range(self.adaptor.getChildCount(t)):
            child = self.adaptor.getChild(t, i)
            self._visitType(child, t, i, ttype, visitor)

    def _visitPattern(self, tree, pattern, visitor):
        """
        For all subtrees that match the pattern, execute the visit action.
        """
        tokenizer = TreePatternLexer(pattern)
        parser = TreePatternParser(tokenizer, self, TreePatternTreeAdaptor())
        tpattern = parser.pattern()
        if tpattern is None or tpattern.isNil() or isinstance(tpattern, WildcardTreePattern):
            return
        rootTokenType = tpattern.getType()

        def rootvisitor(tree, parent, childIndex, labels):
            labels = {}
            if self._parse(tree, tpattern, labels):
                visitor(tree, parent, childIndex, labels)
        self.visit(tree, rootTokenType, rootvisitor)

    def parse(self, t, pattern, labels=None):
        """
        Given a pattern like (ASSIGN %lhs:ID %rhs:.) with optional labels
        on the various nodes and '.' (dot) as the node/subtree wildcard,
        return true if the pattern matches and fill the labels Map with
        the labels pointing at the appropriate nodes.  Return false if
        the pattern is malformed or the tree does not match.

        If a node specifies a text arg in pattern, then that must match
        for that node in t.
        """
        tokenizer = TreePatternLexer(pattern)
        parser = TreePatternParser(tokenizer, self, TreePatternTreeAdaptor())
        tpattern = parser.pattern()
        return self._parse(t, tpattern, labels)

    def _parse(self, t1, t2, labels):
        """
        Do the work for parse. Check to see if the t2 pattern fits the
        structure and token types in t1.  Check text if the pattern has
        text arguments on nodes.  Fill labels map with pointers to nodes
        in tree matched against nodes in pattern with labels.
        """
        if t1 is None or t2 is None:
            return False
        if not isinstance(t2, WildcardTreePattern):
            if self.adaptor.getType(t1) != t2.getType():
                return False
            if t2.hasTextArg and self.adaptor.getText(t1) != t2.getText():
                return False
        if t2.label is not None and labels is not None:
            labels[t2.label] = t1
        n1 = self.adaptor.getChildCount(t1)
        n2 = t2.getChildCount()
        if n1 != n2:
            return False
        for i in range(n1):
            child1 = self.adaptor.getChild(t1, i)
            child2 = t2.getChild(i)
            if not self._parse(child1, child2, labels):
                return False
        return True

    def equals(self, t1, t2, adaptor=None):
        """
        Compare t1 and t2; return true if token types/text, structure match
        exactly.
        The trees are examined in their entirety so that (A B) does not match
        (A B C) nor (A (B C)).
        """
        if adaptor is None:
            adaptor = self.adaptor
        return self._equals(t1, t2, adaptor)

    def _equals(self, t1, t2, adaptor):
        if t1 is None or t2 is None:
            return False
        if adaptor.getType(t1) != adaptor.getType(t2):
            return False
        if adaptor.getText(t1) != adaptor.getText(t2):
            return False
        n1 = adaptor.getChildCount(t1)
        n2 = adaptor.getChildCount(t2)
        if n1 != n2:
            return False
        for i in range(n1):
            child1 = adaptor.getChild(t1, i)
            child2 = adaptor.getChild(t2, i)
            if not self._equals(child1, child2, adaptor):
                return False
        return True