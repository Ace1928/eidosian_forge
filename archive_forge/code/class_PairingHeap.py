from heapq import heappop, heappush
from itertools import count
import networkx as nx
class PairingHeap(MinHeap):
    """A pairing heap."""

    class _Node(MinHeap._Item):
        """A node in a pairing heap.

        A tree in a pairing heap is stored using the left-child, right-sibling
        representation.
        """
        __slots__ = ('left', 'next', 'prev', 'parent')

        def __init__(self, key, value):
            super().__init__(key, value)
            self.left = None
            self.next = None
            self.prev = None
            self.parent = None

    def __init__(self):
        """Initialize a pairing heap."""
        super().__init__()
        self._root = None

    def min(self):
        if self._root is None:
            raise nx.NetworkXError('heap is empty.')
        return (self._root.key, self._root.value)

    def pop(self):
        if self._root is None:
            raise nx.NetworkXError('heap is empty.')
        min_node = self._root
        self._root = self._merge_children(self._root)
        del self._dict[min_node.key]
        return (min_node.key, min_node.value)

    def get(self, key, default=None):
        node = self._dict.get(key)
        return node.value if node is not None else default

    def insert(self, key, value, allow_increase=False):
        node = self._dict.get(key)
        root = self._root
        if node is not None:
            if value < node.value:
                node.value = value
                if node is not root and value < node.parent.value:
                    self._cut(node)
                    self._root = self._link(root, node)
                return True
            elif allow_increase and value > node.value:
                node.value = value
                child = self._merge_children(node)
                if child is not None:
                    self._root = self._link(self._root, child)
            return False
        else:
            node = self._Node(key, value)
            self._dict[key] = node
            self._root = self._link(root, node) if root is not None else node
            return True

    def _link(self, root, other):
        """Link two nodes, making the one with the smaller value the parent of
        the other.
        """
        if other.value < root.value:
            root, other = (other, root)
        next = root.left
        other.next = next
        if next is not None:
            next.prev = other
        other.prev = None
        root.left = other
        other.parent = root
        return root

    def _merge_children(self, root):
        """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
        node = root.left
        root.left = None
        if node is not None:
            link = self._link
            prev = None
            while True:
                next = node.next
                if next is None:
                    node.prev = prev
                    break
                next_next = next.next
                node = link(node, next)
                node.prev = prev
                prev = node
                if next_next is None:
                    break
                node = next_next
            prev = node.prev
            while prev is not None:
                prev_prev = prev.prev
                node = link(prev, node)
                prev = prev_prev
            node.prev = None
            node.next = None
            node.parent = None
        return node

    def _cut(self, node):
        """Cut a node from its parent."""
        prev = node.prev
        next = node.next
        if prev is not None:
            prev.next = next
        else:
            node.parent.left = next
        node.prev = None
        if next is not None:
            next.prev = prev
            node.next = None
        node.parent = None