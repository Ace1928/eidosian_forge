from __future__ import unicode_literals
import re
class NodeWalker(object):

    def __init__(self, root):
        self.current = root
        self.root = root
        self.entering = True

    def __next__(self):
        cur = self.current
        entering = self.entering
        if cur is None:
            raise StopIteration
        container = is_container(cur)
        if entering and container:
            if cur.first_child:
                self.current = cur.first_child
                self.entering = True
            else:
                self.entering = False
        elif cur == self.root:
            self.current = None
        elif cur.nxt is None:
            self.current = cur.parent
            self.entering = False
        else:
            self.current = cur.nxt
            self.entering = True
        return (cur, entering)
    next = __next__

    def __iter__(self):
        return self

    def nxt(self):
        """ for backwards compatibility """
        try:
            cur, entering = next(self)
            return {'entering': entering, 'node': cur}
        except StopIteration:
            return None

    def resume_at(self, node, entering):
        self.current = node
        self.entering = entering is True