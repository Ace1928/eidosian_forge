import abc
import collections
import collections.abc
import operator
import sys
import typing
def _get_cons(self):
    """Return the class used to create instance of this type."""
    if self.__origin__ is None:
        raise TypeError('Cannot get the underlying type of a non-specialized Annotated type.')
    tree = self._subs_tree()
    while isinstance(tree, tuple) and tree[0] is Annotated:
        tree = tree[1]
    if isinstance(tree, tuple):
        return tree[0]
    else:
        return tree