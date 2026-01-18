from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
def _submit_feedback(self, key: str, results: dict):
    current_run = rh.get_current_run_tree()
    run_id = current_run.id if current_run else None
    if not ls_utils.test_tracking_is_disabled():
        self.executor.submit(self.client.create_feedback, run_id=run_id, key=key, **results)