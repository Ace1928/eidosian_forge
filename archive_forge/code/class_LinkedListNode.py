import weakref
from weakref import ReferenceType
class LinkedListNode(Generic[T]):
    __slots__ = ('_previous_node', 'value', 'next_node', '__weakref__')

    def __init__(self, value):
        self._previous_node = None
        self.next_node = None
        self.value = value

    @property
    def previous_node(self):
        return resolve_ref(self._previous_node)

    @previous_node.setter
    def previous_node(self, node):
        self._previous_node = weakref.ref(node) if node is not None else None

    def remove(self):
        LinkedListNode.link_nodes(self.previous_node, self.next_node)
        self.previous_node = None
        self.next_node = None
        return self.value

    def iter_next(self, *, skip_current=False):
        node = self.next_node if skip_current else self
        while node:
            yield node
            node = node.next_node

    def iter_previous(self, *, skip_current=False):
        node = self.previous_node if skip_current else self
        while node:
            yield node
            node = node.previous_node

    @staticmethod
    def link_nodes(previous_node, next_node):
        if next_node:
            next_node.previous_node = previous_node
        if previous_node:
            previous_node.next_node = next_node

    @staticmethod
    def _insert_link(first_node, new_node, last_node):
        LinkedListNode.link_nodes(first_node, new_node)
        LinkedListNode.link_nodes(new_node, last_node)

    def insert_before(self, new_node):
        assert self is not new_node and new_node is not self.previous_node
        LinkedListNode._insert_link(self.previous_node, new_node, self)

    def insert_after(self, new_node):
        assert self is not new_node and new_node is not self.next_node
        LinkedListNode._insert_link(self, new_node, self.next_node)