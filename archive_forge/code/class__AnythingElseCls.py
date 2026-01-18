from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
@total_ordering
class _AnythingElseCls:
    """
        This is a surrogate symbol which you can use in your finite state machines
        to represent "any symbol not in the official alphabet". For example, if your
        state machine's alphabet is {"a", "b", "c", "d", fsm.anything_else}, then
        you can pass "e" in as a symbol and it will be converted to
        fsm.anything_else, then follow the appropriate transition.
    """

    def __str__(self):
        return 'anything_else'

    def __repr__(self):
        return 'anything_else'

    def __lt__(self, other):
        return False

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))