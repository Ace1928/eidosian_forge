import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _verify_accept_kwargs(self, func):
    """Verifies a callable accepts kwargs

        :type func: callable
        :param func: A callable object.

        :returns: True, if ``func`` accepts kwargs, otherwise False.

        """
    try:
        if not accepts_kwargs(func):
            raise ValueError(f'Event handler {func} must accept keyword arguments (**kwargs)')
    except TypeError:
        return False