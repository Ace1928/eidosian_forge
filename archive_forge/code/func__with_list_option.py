import itertools
import operator
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import MutableSequence
from copy import deepcopy
from functools import partial as _partial
from functools import reduce
from operator import itemgetter
from types import GeneratorType
from kombu.utils.functional import fxrange, reprcall
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import barrier
from celery._state import current_app
from celery.exceptions import CPendingDeprecationWarning
from celery.result import GroupResult, allow_join_result
from celery.utils import abstract
from celery.utils.collections import ChainMap
from celery.utils.functional import _regen
from celery.utils.functional import chunks as _chunks
from celery.utils.functional import is_list, maybe_list, regen, seq_concat_item, seq_concat_seq
from celery.utils.objects import getitem_property
from celery.utils.text import remove_repeating_from_task, truncate
def _with_list_option(self, key):
    """Gets the value at the given self.options[key] as a list.

        If the value is not a list, it will be converted to one and saved in self.options.
        If the key does not exist, an empty list will be set and returned instead.

        Arguments:
            key (str): The key to get the value for.

        Returns:
            List: The value at the given key as a list or an empty list if the key does not exist.
        """
    items = self.options.setdefault(key, [])
    if not isinstance(items, MutableSequence):
        items = self.options[key] = [items]
    return items