from __future__ import annotations
from collections import deque
from dask.core import istask, subs
class Traverser:
    """Traverser interface for tasks.

    Class for storing the state while performing a preorder-traversal of a
    task.

    Parameters
    ----------
    term : task
        The task to be traversed

    Attributes
    ----------
    term
        The current element in the traversal
    current
        The head of the current element in the traversal. This is simply `head`
        applied to the attribute `term`.
    """

    def __init__(self, term, stack=None):
        self.term = term
        if not stack:
            self._stack = deque([END])
        else:
            self._stack = stack

    def __iter__(self):
        while self.current is not END:
            yield self.current
            self.next()

    def copy(self):
        """Copy the traverser in its current state.

        This allows the traversal to be pushed onto a stack, for easy
        backtracking."""
        return Traverser(self.term, deque(self._stack))

    def next(self):
        """Proceed to the next term in the preorder traversal."""
        subterms = args(self.term)
        if not subterms:
            self.term = self._stack.pop()
        else:
            self.term = subterms[0]
            self._stack.extend(reversed(subterms[1:]))

    @property
    def current(self):
        return head(self.term)

    def skip(self):
        """Skip over all subterms of the current level in the traversal"""
        self.term = self._stack.pop()