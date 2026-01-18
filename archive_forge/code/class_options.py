import functools
import logging
from typing import Dict, Set, List, Tuple, Union, Optional, Any
import time
import uuid
import ray
from ray.dag import DAGNode
from ray.dag.input_node import DAGInputData
from ray.remote_function import RemoteFunction
from ray.workflow.common import (
from ray.workflow import serialization, workflow_access, workflow_context
from ray.workflow.event_listener import EventListener, EventListenerType, TimerListener
from ray.workflow.workflow_storage import WorkflowStorage
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
from ray.util.annotations import PublicAPI
from ray._private.usage import usage_lib
@PublicAPI(stability='alpha')
class options:
    """This class serves both as a decorator and options for workflow.

    Examples:

        .. testcode::

            import ray
            from ray import workflow

            # specify workflow options with a decorator
            @workflow.options(catch_exceptions=True)
            @ray.remote
            def foo():
                return 1

            # specify workflow options in ".options"
            foo_new = foo.options(**workflow.options(catch_exceptions=False))
    """

    def __init__(self, **workflow_options: Dict[str, Any]):
        valid_options = {'task_id', 'metadata', 'catch_exceptions', 'checkpoint'}
        invalid_keywords = set(workflow_options.keys()) - valid_options
        if invalid_keywords:
            raise ValueError(f'Invalid option keywords {invalid_keywords} for workflow tasks. Valid ones are {valid_options}.')
        from ray.workflow.common import WORKFLOW_OPTIONS
        validate_user_metadata(workflow_options.get('metadata'))
        self.options = {'_metadata': {WORKFLOW_OPTIONS: workflow_options}}

    def keys(self):
        return ('_metadata',)

    def __getitem__(self, key):
        return self.options[key]

    def __call__(self, f: RemoteFunction) -> RemoteFunction:
        if not isinstance(f, RemoteFunction):
            raise ValueError("Only apply 'workflow.options' to Ray remote functions.")
        f._default_options.update(self.options)
        return f