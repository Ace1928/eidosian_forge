from suds import *
from suds.sax import splitPrefix, Namespace
from suds.sudsobject import Object
from suds.xsd.query import BlindQuery, TypeQuery, qualify
import re
from logging import getLogger
class TreeResolver(Resolver):
    """
    The tree resolver is a I{stateful} tree resolver
    used to resolve each node in a tree.  As such, it mirrors
    the tree structure to ensure that nodes are resolved in
    context.
    @ivar stack: The context stack.
    @type stack: list
    """

    def __init__(self, schema):
        """
        @param schema: A schema object.
        @type schema: L{xsd.schema.Schema}
        """
        Resolver.__init__(self, schema)
        self.stack = Stack()

    def reset(self):
        """
        Reset the resolver's state.
        """
        self.stack = Stack()

    def push(self, x):
        """
        Push an I{object} onto the stack.
        @param x: An object to push.
        @type x: L{Frame}
        @return: The pushed frame.
        @rtype: L{Frame}
        """
        if isinstance(x, Frame):
            frame = x
        else:
            frame = Frame(x)
        self.stack.append(frame)
        log.debug('push: (%s)\n%s', Repr(frame), Repr(self.stack))
        return frame

    def top(self):
        """
        Get the I{frame} at the top of the stack.
        @return: The top I{frame}, else Frame.Empty.
        @rtype: L{Frame}
        """
        if len(self.stack):
            return self.stack[-1]
        else:
            return Frame.Empty()

    def pop(self):
        """
        Pop the frame at the top of the stack.
        @return: The popped frame, else None.
        @rtype: L{Frame}
        """
        if len(self.stack):
            popped = self.stack.pop()
            log.debug('pop: (%s)\n%s', Repr(popped), Repr(self.stack))
            return popped
        log.debug('stack empty, not-popped')
        return None

    def depth(self):
        """
        Get the current stack depth.
        @return: The current stack depth.
        @rtype: int
        """
        return len(self.stack)

    def getchild(self, name, parent):
        """Get a child by name."""
        log.debug('searching parent (%s) for (%s)', Repr(parent), name)
        if name.startswith('@'):
            return parent.get_attribute(name[1:])
        return parent.get_child(name)