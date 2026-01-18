import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def _get_new_training_session_class():
    """
    Returns a session manager class for nested autologging runs.

    Examples
    --------
    >>> class Parent:
    ...     pass
    >>> class Child:
    ...     pass
    >>> class Grandchild:
    ...     pass
    >>>
    >>> _TrainingSession = _get_new_training_session_class()
    >>> with _TrainingSession(Parent, False) as p:
    ...     with _SklearnTrainingSession(Child, True) as c:
    ...         with _SklearnTrainingSession(Grandchild, True) as g:
    ...             print(p.should_log(), c.should_log(), g.should_log())
    True False False
    >>>
    >>> with _TrainingSession(Parent, True) as p:
    ...     with _TrainingSession(Child, False) as c:
    ...         with _TrainingSession(Grandchild, True) as g:
    ...             print(p.should_log(), c.should_log(), g.should_log())
    True True False
    >>>
    >>> with _TrainingSession(Child, True) as c1:
    ...     with _TrainingSession(Child, True) as c2:
    ...         print(c1.should_log(), c2.should_log())
    True False
    """

    class _TrainingSession:
        _session_stack = []

        def __init__(self, estimator, allow_children=True):
            """A session manager for nested autologging runs.

            Args:
                estimator: An estimator that this session originates from.
                allow_children: If True, allows autologging in child sessions.
                    If False, disallows autologging in all descendant sessions.

            """
            self.allow_children = allow_children
            self.estimator = estimator
            self._parent = None

        def __enter__(self):
            if len(_TrainingSession._session_stack) > 0:
                self._parent = _TrainingSession._session_stack[-1]
                self.allow_children = _TrainingSession._session_stack[-1].allow_children and self.allow_children
            _TrainingSession._session_stack.append(self)
            return self

        def __exit__(self, tp, val, traceback):
            _TrainingSession._session_stack.pop()

        def should_log(self):
            """
            Returns True when at least one of the following conditions satisfies:

            1. This session is the root session.
            2. The parent session allows autologging and its estimator differs from this session's
               estimator.
            """
            for training_session in _TrainingSession._session_stack:
                if training_session is self:
                    break
                elif training_session.estimator is self.estimator:
                    return False
            return self._parent is None or self._parent.allow_children

        @staticmethod
        def is_active():
            return len(_TrainingSession._session_stack) != 0

        @staticmethod
        def get_current_session():
            if _TrainingSession.is_active():
                return _TrainingSession._session_stack[-1]
            return None
    return _TrainingSession