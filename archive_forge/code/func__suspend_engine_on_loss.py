import logging
import os
from taskflow import exceptions
from taskflow.listeners import base
from taskflow import states
def _suspend_engine_on_loss(self, engine, state, details):
    """The default strategy for handling claims being lost."""
    try:
        engine.suspend()
    except exceptions.TaskFlowException as e:
        LOG.warning("Failed suspending engine '%s', (previously owned by '%s'):%s%s", engine, self._owner, os.linesep, e.pformat())