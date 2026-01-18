import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _verify_is_callable(self, func):
    if not callable(func):
        raise ValueError('Event handler %s must be callable.' % func)