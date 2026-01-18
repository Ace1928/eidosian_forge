import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
class WorkflowIndexingStorage:
    """Access and maintenance the indexing of workflow status.

    It runs a protocol that guarantees we can recover from any interrupted
    status updating. This protocol is **not thread-safe** for updating the
    status of the same workflow, currently it is executed by workflow management
    actor with a single thread.

    Here is how the protocol works:

    Update the status of a workflow
    1. Load workflow status from workflow data. If it is the same as the new status,
       return.
    2. Check if the workflow status updating is dirty. If it is, fix the
       workflow status; otherwise, mark the workflow status updating dirty.
    3. Update status in the workflow metadata.
    4. Insert the workflow ID key in the status indexing directory of the new status.
    5. Delete the workflow ID key in the status indexing directory of
       the previous status.
    6. Remove the workflow status updating dirty mark.

    Load a status of a workflow
    1. Read the status of the workflow from the workflow metadata.
    2. Return the status.

    List the status of all workflows
    1. Get status of all workflows by listing workflow ID keys in each workflow
       status indexing directory.
    2. List all workflows with dirty updating status. Get their status from
       workflow data. Override the status of the corresponding workflow.
    3. Return all the status.
    """

    def __init__(self):
        self._storage = storage.get_client(WORKFLOW_ROOT)

    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus):
        """Update the status of the workflow.
        Try fixing indexing if workflow status updating was marked dirty.

        This method is NOT thread-safe. It is handled by the workflow management actor.
        """
        prev_status = self.load_workflow_status(workflow_id)
        if prev_status != status:
            if self._storage.get_info(self._key_workflow_status_dirty(workflow_id)) is not None:
                self._storage.put(self._key_workflow_with_status(workflow_id, prev_status), b'')
                for s in WorkflowStatus:
                    if s != prev_status:
                        self._storage.delete(self._key_workflow_with_status(workflow_id, s))
            else:
                self._storage.put(self._key_workflow_status_dirty(workflow_id), b'')
            self._storage.put(self._key_workflow_metadata(workflow_id), json.dumps({'status': status.value}).encode())
            self._storage.put(self._key_workflow_with_status(workflow_id, status), b'')
            if prev_status is not WorkflowStatus.NONE:
                self._storage.delete(self._key_workflow_with_status(workflow_id, prev_status))
            self._storage.delete(self._key_workflow_status_dirty(workflow_id))

    def load_workflow_status(self, workflow_id: str):
        """Load the committed workflow status."""
        raw_data = self._storage.get(self._key_workflow_metadata(workflow_id))
        if raw_data is not None:
            metadata = json.loads(raw_data)
            return WorkflowStatus(metadata['status'])
        return WorkflowStatus.NONE

    def list_workflow(self, status_filter: Optional[Set[WorkflowStatus]]=None) -> List[Tuple[str, WorkflowStatus]]:
        """List workflow status. Override status of the workflows whose status updating
        were marked dirty with the workflow status from workflow metadata.

        Args:
            status_filter: If given, only returns workflow with that status. This can
                be a single status or set of statuses.
        """
        if status_filter is None:
            status_filter = set(WorkflowStatus)
            status_filter.discard(WorkflowStatus.NONE)
        elif not isinstance(status_filter, set):
            raise TypeError("'status_filter' should either be 'None' or a set.")
        elif WorkflowStatus.NONE in status_filter:
            raise ValueError("'WorkflowStatus.NONE' is not a valid filter value.")
        results = {}
        for status in status_filter:
            try:
                for p in self._storage.list(self._key_workflow_with_status('', status)):
                    workflow_id = p.base_name
                    results[workflow_id] = status
            except FileNotFoundError:
                pass
        try:
            for p in self._storage.list(self._key_workflow_status_dirty('')):
                workflow_id = p.base_name
                results.pop(workflow_id, None)
                status = self.load_workflow_status(workflow_id)
                if status in status_filter:
                    results[workflow_id] = status
        except FileNotFoundError:
            pass
        return list(results.items())

    def delete_workflow_status(self, workflow_id: str):
        """Delete status indexing for the workflow."""
        for status in WorkflowStatus:
            self._storage.delete(self._key_workflow_with_status(workflow_id, status))
        self._storage.delete(self._key_workflow_status_dirty(workflow_id))

    def _key_workflow_with_status(self, workflow_id: str, status: WorkflowStatus):
        """A key whose existence marks the status of the workflow."""
        return os.path.join(WORKFLOW_STATUS_DIR, status.value, workflow_id)

    def _key_workflow_status_dirty(self, workflow_id: str):
        """A key marks the workflow status dirty, because it is under change."""
        return os.path.join(WORKFLOW_STATUS_DIR, WORKFLOW_STATUS_DIRTY_DIR, workflow_id)

    def _key_workflow_metadata(self, workflow_id: str):
        return os.path.join(workflow_id, WORKFLOW_META)