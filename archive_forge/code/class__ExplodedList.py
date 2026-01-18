from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
class _ExplodedList(list):
    """
    Wrapper around a list, that marks it as 'exploded'.

    As soon as items are added or the list is extended, the new items are
    automatically exploded as well.
    """

    def __init__(self, *a, **kw):
        super(_ExplodedList, self).__init__(*a, **kw)
        self.exploded = True

    def append(self, item):
        self.extend([item])

    def extend(self, lst):
        super(_ExplodedList, self).extend(explode_tokens(lst))

    def insert(self, index, item):
        raise NotImplementedError

    def __setitem__(self, index, value):
        """
        Ensure that when `(Token, 'long string')` is set, the string will be
        exploded.
        """
        if not isinstance(index, slice):
            index = slice(index, index + 1)
        value = explode_tokens([value])
        super(_ExplodedList, self).__setitem__(index, value)