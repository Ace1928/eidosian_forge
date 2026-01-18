import asyncio
from typing import Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import ray
from ray import serve
from ray.workflow import common, workflow_context, workflow_access
from ray.workflow.event_listener import EventListener
from ray.workflow.common import Event
import logging
class WorkflowEventHandleError(Exception):
    """Raise when event processing failed"""

    def __init__(self, workflow_id: str, what_happened: str):
        self.message = f'Workflow[id={workflow_id}] HTTP event handle failed: {what_happened}'
        super().__init__(self.message)