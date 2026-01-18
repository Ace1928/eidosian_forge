import abc
from oslo_utils import excutils
from taskflow import logging
from taskflow import states
from taskflow.types import failure
from taskflow.types import notifier
class DumpingListener(Listener, metaclass=abc.ABCMeta):
    """Abstract base class for dumping listeners.

    This provides a simple listener that can be attached to an engine which can
    be derived from to dump task and/or flow state transitions to some target
    backend.

    To implement your own dumping listener derive from this class and
    override the ``_dump`` method.
    """

    @abc.abstractmethod
    def _dump(self, message, *args, **kwargs):
        """Dumps the provided *templated* message to some output."""

    def _flow_receiver(self, state, details):
        self._dump("%s has moved flow '%s' (%s) into state '%s' from state '%s'", self._engine, details['flow_name'], details['flow_uuid'], state, details['old_state'])

    def _task_receiver(self, state, details):
        if state in FINISH_STATES:
            result = details.get('result')
            exc_info = None
            was_failure = False
            if isinstance(result, failure.Failure):
                if result.exc_info:
                    exc_info = tuple(result.exc_info)
                was_failure = True
            self._dump("%s has moved task '%s' (%s) into state '%s' from state '%s' with result '%s' (failure=%s)", self._engine, details['task_name'], details['task_uuid'], state, details['old_state'], result, was_failure, exc_info=exc_info)
        else:
            self._dump("%s has moved task '%s' (%s) into state '%s' from state '%s'", self._engine, details['task_name'], details['task_uuid'], state, details['old_state'])