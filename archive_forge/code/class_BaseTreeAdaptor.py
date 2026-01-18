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
class BaseTreeAdaptor(TreeAdaptor):
    """
    @brief A TreeAdaptor that works with any Tree implementation.
    """

    def nil(self):
        return self.createWithPayload(None)

    def errorNode(self, input, start, stop, exc):
        """
        create tree node that holds the start and stop tokens associated
        with an error.

        If you specify your own kind of tree nodes, you will likely have to
        override this method. CommonTree returns Token.INVALID_TOKEN_TYPE
        if no token payload but you might have to set token type for diff
        node type.
        """
        return CommonErrorNode(input, start, stop, exc)

    def isNil(self, tree):
        return tree.isNil()

    def dupTree(self, t, parent=None):
        """
        This is generic in the sense that it will work with any kind of
        tree (not just Tree interface).  It invokes the adaptor routines
        not the tree node routines to do the construction.
        """
        if t is None:
            return None
        newTree = self.dupNode(t)
        self.setChildIndex(newTree, self.getChildIndex(t))
        self.setParent(newTree, parent)
        for i in range(self.getChildCount(t)):
            child = self.getChild(t, i)
            newSubTree = self.dupTree(child, t)
            self.addChild(newTree, newSubTree)
        return newTree

    def addChild(self, tree, child):
        """
        Add a child to the tree t.  If child is a flat tree (a list), make all
        in list children of t.  Warning: if t has no children, but child does
        and child isNil then you can decide it is ok to move children to t via
        t.children = child.children; i.e., without copying the array.  Just
        make sure that this is consistent with have the user will build
        ASTs.
        """
        if tree is not None and child is not None:
            tree.addChild(child)

    def becomeRoot(self, newRoot, oldRoot):
        """
        If oldRoot is a nil root, just copy or move the children to newRoot.
        If not a nil root, make oldRoot a child of newRoot.

          old=^(nil a b c), new=r yields ^(r a b c)
          old=^(a b c), new=r yields ^(r ^(a b c))

        If newRoot is a nil-rooted single child tree, use the single
        child as the new root node.

          old=^(nil a b c), new=^(nil r) yields ^(r a b c)
          old=^(a b c), new=^(nil r) yields ^(r ^(a b c))

        If oldRoot was null, it's ok, just return newRoot (even if isNil).

          old=null, new=r yields r
          old=null, new=^(nil r) yields ^(nil r)

        Return newRoot.  Throw an exception if newRoot is not a
        simple node or nil root with a single child node--it must be a root
        node.  If newRoot is ^(nil x) return x as newRoot.

        Be advised that it's ok for newRoot to point at oldRoot's
        children; i.e., you don't have to copy the list.  We are
        constructing these nodes so we should have this control for
        efficiency.
        """
        if isinstance(newRoot, Token):
            newRoot = self.create(newRoot)
        if oldRoot is None:
            return newRoot
        if not isinstance(newRoot, CommonTree):
            newRoot = self.createWithPayload(newRoot)
        if newRoot.isNil():
            nc = newRoot.getChildCount()
            if nc == 1:
                newRoot = newRoot.getChild(0)
            elif nc > 1:
                raise RuntimeError('more than one node as root')
        newRoot.addChild(oldRoot)
        return newRoot

    def rulePostProcessing(self, root):
        """Transform ^(nil x) to x and nil to null"""
        if root is not None and root.isNil():
            if root.getChildCount() == 0:
                root = None
            elif root.getChildCount() == 1:
                root = root.getChild(0)
                root.setParent(None)
                root.setChildIndex(-1)
        return root

    def createFromToken(self, tokenType, fromToken, text=None):
        assert isinstance(tokenType, six.integer_types), type(tokenType).__name__
        assert isinstance(fromToken, Token), type(fromToken).__name__
        assert text is None or isinstance(text, six.string_types), type(text).__name__
        fromToken = self.createToken(fromToken)
        fromToken.type = tokenType
        if text is not None:
            fromToken.text = text
        t = self.createWithPayload(fromToken)
        return t

    def createFromType(self, tokenType, text):
        assert isinstance(tokenType, six.integer_types), type(tokenType).__name__
        assert isinstance(text, six.string_types), type(text).__name__
        fromToken = self.createToken(tokenType=tokenType, text=text)
        t = self.createWithPayload(fromToken)
        return t

    def getType(self, t):
        return t.getType()

    def setType(self, t, type):
        raise RuntimeError("don't know enough about Tree node")

    def getText(self, t):
        return t.getText()

    def setText(self, t, text):
        raise RuntimeError("don't know enough about Tree node")

    def getChild(self, t, i):
        return t.getChild(i)

    def setChild(self, t, i, child):
        t.setChild(i, child)

    def deleteChild(self, t, i):
        return t.deleteChild(i)

    def getChildCount(self, t):
        return t.getChildCount()

    def getUniqueID(self, node):
        return hash(node)

    def createToken(self, fromToken=None, tokenType=None, text=None):
        """
        Tell me how to create a token for use with imaginary token nodes.
        For example, there is probably no input symbol associated with imaginary
        token DECL, but you need to create it as a payload or whatever for
        the DECL node as in ^(DECL type ID).

        If you care what the token payload objects' type is, you should
        override this method and any other createToken variant.
        """
        raise NotImplementedError