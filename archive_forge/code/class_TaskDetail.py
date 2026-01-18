import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
class TaskDetail(AtomDetail):
    """A task detail (an atom detail typically associated with a |tt| atom).

    .. |tt| replace:: :py:class:`~taskflow.task.Task`
    """

    def reset(self, state):
        """Resets this task detail and sets ``state`` attribute value.

        This sets any previously set ``results``, ``failure``,
        and ``revert_results`` attributes back to ``None`` and sets the
        state to the provided one, as well as setting this task
        details ``intention`` attribute to ``EXECUTE``.
        """
        self.results = None
        self.failure = None
        self.revert_results = None
        self.revert_failure = None
        self.state = state
        self.intention = states.EXECUTE

    def put(self, state, result):
        """Puts a result (acquired in the given state) into this detail.

        Returns whether this object was modified (or whether it was not).
        """
        was_altered = False
        if state != self.state:
            self.state = state
            was_altered = True
        if state == states.REVERT_FAILURE:
            if self.revert_failure != result:
                self.revert_failure = result
                was_altered = True
            if not _is_all_none(self.results, self.revert_results):
                self.results = None
                self.revert_results = None
                was_altered = True
        elif state == states.FAILURE:
            if self.failure != result:
                self.failure = result
                was_altered = True
            if not _is_all_none(self.results, self.revert_results, self.revert_failure):
                self.results = None
                self.revert_results = None
                self.revert_failure = None
                was_altered = True
        elif state == states.SUCCESS:
            if not _is_all_none(self.revert_results, self.revert_failure, self.failure):
                self.revert_results = None
                self.revert_failure = None
                self.failure = None
                was_altered = True
            if result is not self.results:
                self.results = result
                was_altered = True
        elif state == states.REVERTED:
            if not _is_all_none(self.revert_failure):
                self.revert_failure = None
                was_altered = True
            if result is not self.revert_results:
                self.revert_results = result
                was_altered = True
        return was_altered

    def merge(self, other, deep_copy=False):
        """Merges the current task detail with the given one.

        NOTE(harlowja): This merge does **not** copy and replace
        the ``results`` or ``revert_results`` if it differs. Instead the
        current objects ``results`` and ``revert_results`` attributes directly
        becomes (via assignment) the other objects attributes. Also note that
        if the provided object is this object itself then **no** merging is
        done.

        See: https://bugs.launchpad.net/taskflow/+bug/1452978 for
        what happens if this is copied at a deeper level (for example by
        using ``copy.deepcopy`` or by using ``copy.copy``).

        :returns: this task detail (freshly merged with the incoming object)
        :rtype: :py:class:`.TaskDetail`
        """
        if not isinstance(other, TaskDetail):
            raise exc.NotImplementedError('Can only merge with other task details')
        if other is self:
            return self
        super(TaskDetail, self).merge(other, deep_copy=deep_copy)
        self.results = other.results
        self.revert_results = other.revert_results
        return self

    def copy(self):
        """Copies this task detail.

        Creates a shallow copy of this task detail (any meta-data and
        version information that this object maintains is shallow
        copied via ``copy.copy``).

        NOTE(harlowja): This copy does **not** copy and replace
        the ``results`` or ``revert_results`` attribute if it differs. Instead
        the current objects ``results`` and ``revert_results`` attributes
        directly becomes (via assignment) the cloned objects attributes.

        See: https://bugs.launchpad.net/taskflow/+bug/1452978 for
        what happens if this is copied at a deeper level (for example by
        using ``copy.deepcopy`` or by using ``copy.copy``).

        :returns: a new task detail
        :rtype: :py:class:`.TaskDetail`
        """
        clone = copy.copy(self)
        clone.results = self.results
        clone.revert_results = self.revert_results
        if self.meta:
            clone.meta = self.meta.copy()
        if self.version:
            clone.version = copy.copy(self.version)
        return clone