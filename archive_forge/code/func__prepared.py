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
def _prepared(self, tasks, partial_args, group_id, root_id, app, CallableSignature=abstract.CallableSignature, from_dict=Signature.from_dict, isinstance=isinstance, tuple=tuple):
    """Recursively unroll the group into a generator of its tasks.

        This is used by :meth:`apply_async` and :meth:`apply` to
        unroll the group into a list of tasks that can be evaluated.

        Note:
            This does not change the group itself, it only returns
            a generator of the tasks that the group would evaluate to.

        Arguments:
            tasks (list): List of tasks in the group (may contain nested groups).
            partial_args (list): List of arguments to be prepended to
                the arguments of each task.
            group_id (str): The group id of the group.
            root_id (str): The root id of the group.
            app (Celery): The Celery app instance.
            CallableSignature (class): The signature class of the group's tasks.
            from_dict (fun): Function to create a signature from a dict.
            isinstance (fun): Function to check if an object is an instance
                of a class.
            tuple (class): A tuple-like class.

        Returns:
            generator: A generator for the unrolled group tasks.
                The generator yields tuples of the form ``(task, AsyncResult, group_id)``.
        """
    for index, task in enumerate(tasks):
        if isinstance(task, CallableSignature):
            task = task.clone()
        else:
            task = from_dict(task, app=app)
        if isinstance(task, group):
            unroll = task._prepared(task.tasks, partial_args, group_id, root_id, app)
            yield from unroll
        else:
            if partial_args and (not task.immutable):
                task.args = tuple(partial_args) + tuple(task.args)
            yield (task, task.freeze(group_id=group_id, root_id=root_id, group_index=index), group_id)