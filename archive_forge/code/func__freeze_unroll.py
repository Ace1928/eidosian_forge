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
def _freeze_unroll(self, new_tasks, group_id, chord, root_id, parent_id):
    """Generator for the frozen flattened group tasks.

        Creates a flattened list of the tasks in the group, and freezes
        each task in the group. Nested groups will be recursively flattened.

        Exhausting the generator will create a new list of the flattened
        tasks in the group and will return it in the new_tasks argument.

        Arguments:
            new_tasks (list): The list to append the flattened tasks to.
            group_id (str): The group_id to use for the tasks.
            chord (Chord): The chord to use for the tasks.
            root_id (str): The root_id to use for the tasks.
            parent_id (str): The parent_id to use for the tasks.

        Yields:
            AsyncResult: The frozen task.
        """
    stack = deque(self.tasks)
    group_index = 0
    while stack:
        task = maybe_signature(stack.popleft(), app=self._app).clone()
        if isinstance(task, group):
            stack.extendleft(task.tasks)
        else:
            new_tasks.append(task)
            yield task.freeze(group_id=group_id, chord=chord, root_id=root_id, parent_id=parent_id, group_index=group_index)
            group_index += 1