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
def _freeze_group_tasks(self, _id=None, group_id=None, chord=None, root_id=None, parent_id=None, group_index=None):
    """Freeze the tasks in the group.

        Note:
            If the group tasks are created from a generator, the tasks generator would
            not be exhausted, and the tasks would be frozen lazily.

        Returns:
            tuple: A tuple of the group id, and the AsyncResult of each of the group tasks.
        """
    opts = self.options
    try:
        gid = opts['task_id']
    except KeyError:
        gid = opts['task_id'] = group_id or uuid()
    if group_id:
        opts['group_id'] = group_id
    if chord:
        opts['chord'] = chord
    if group_index is not None:
        opts['group_index'] = group_index
    root_id = opts.setdefault('root_id', root_id)
    parent_id = opts.setdefault('parent_id', parent_id)
    if isinstance(self.tasks, _regen):
        tasks1, tasks2 = itertools.tee(self._unroll_tasks(self.tasks))
        results = regen(self._freeze_tasks(tasks1, group_id, chord, root_id, parent_id))
        self.tasks = regen(tasks2)
    else:
        new_tasks = []
        results = list(self._freeze_unroll(new_tasks, group_id, chord, root_id, parent_id))
        if isinstance(self.tasks, MutableSequence):
            self.tasks[:] = new_tasks
        else:
            self.tasks = new_tasks
    return (gid, results)