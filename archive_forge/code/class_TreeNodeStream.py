from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
class TreeNodeStream(IntStream):
    """@brief A stream of tree nodes

    It accessing nodes from a tree of some kind.
    """

    def get(self, i):
        """Get a tree node at an absolute index i; 0..n-1.

        If you don't want to buffer up nodes, then this method makes no
        sense for you.
        """
        raise NotImplementedError

    def LT(self, k):
        """
        Get tree node at current input pointer + i ahead where i=1 is next node.
        i<0 indicates nodes in the past.  So LT(-1) is previous node, but
        implementations are not required to provide results for k < -1.
        LT(0) is undefined.  For i>=n, return null.
        Return null for LT(0) and any index that results in an absolute address
        that is negative.

        This is analogus to the LT() method of the TokenStream, but this
        returns a tree node instead of a token.  Makes code gen identical
        for both parser and tree grammars. :)
        """
        raise NotImplementedError

    def getTreeSource(self):
        """
        Where is this stream pulling nodes from?  This is not the name, but
        the object that provides node objects.
        """
        raise NotImplementedError

    def getTokenStream(self):
        """
        If the tree associated with this stream was created from a TokenStream,
        you can specify it here.  Used to do rule $text attribute in tree
        parser.  Optional unless you use tree parser rule text attribute
        or output=template and rewrite=true options.
        """
        raise NotImplementedError

    def getTreeAdaptor(self):
        """
        What adaptor can tell me how to interpret/navigate nodes and
        trees.  E.g., get text of a node.
        """
        raise NotImplementedError

    def setUniqueNavigationNodes(self, uniqueNavigationNodes):
        """
        As we flatten the tree, we use UP, DOWN nodes to represent
        the tree structure.  When debugging we need unique nodes
        so we have to instantiate new ones.  When doing normal tree
        parsing, it's slow and a waste of memory to create unique
        navigation nodes.  Default should be false;
        """
        raise NotImplementedError

    def toString(self, start, stop):
        """
        Return the text of all nodes from start to stop, inclusive.
        If the stream does not buffer all the nodes then it can still
        walk recursively from start until stop.  You can always return
        null or "" too, but users should not access $ruleLabel.text in
        an action of course in that case.
        """
        raise NotImplementedError

    def replaceChildren(self, parent, startChildIndex, stopChildIndex, t):
        """
        Replace from start to stop child index of parent with t, which might
        be a list.  Number of children may be different
        after this call.  The stream is notified because it is walking the
        tree and might need to know you are monkeying with the underlying
        tree.  Also, it might be able to modify the node stream to avoid
        restreaming for future phases.

        If parent is null, don't do anything; must be at root of overall tree.
        Can't replace whatever points to the parent externally.  Do nothing.
        """
        raise NotImplementedError