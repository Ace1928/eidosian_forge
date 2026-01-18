from pprint import pformat
from .py3compat import MutableMapping
class ListContainer(list):
    """
    A container for lists.
    """
    __slots__ = ['__recursion_lock__']

    @recursion_lock('[...]')
    def __str__(self):
        return pformat(self)