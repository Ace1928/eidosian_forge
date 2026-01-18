import numpy as np
class DrawTree:

    def __init__(self, tree, parent=None, depth=0, number=1):
        self.x = -1.0
        self.y = depth
        self.tree = tree
        self.children = [DrawTree(c, self, depth + 1, i + 1) for i, c in enumerate(tree.children)]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        self.number = number

    def left(self):
        return self.thread or (len(self.children) and self.children[0])

    def right(self):
        return self.thread or (len(self.children) and self.children[-1])

    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                else:
                    n = node
        return n

    def get_lmost_sibling(self):
        if not self._lmost_sibling and self.parent and (self != self.parent.children[0]):
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling
    lmost_sibling = property(get_lmost_sibling)

    def __str__(self):
        return '%s: x=%s mod=%s' % (self.tree, self.x, self.mod)

    def __repr__(self):
        return self.__str__()

    def max_extents(self):
        extents = [c.max_extents() for c in self.children]
        extents.append((self.x, self.y))
        return np.max(extents, axis=0)