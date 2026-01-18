import abc
import os
import threading
import fasteners
from taskflow import engines
from taskflow import exceptions as excp
from taskflow.types import entity
from taskflow.types import notifier
from taskflow.utils import misc
def _engine_from_job(self, job):
    """Extracts an engine from a job (via some manner)."""
    flow_detail = self._flow_detail_from_job(job)
    store = {}
    if flow_detail.meta and 'store' in flow_detail.meta:
        store.update(flow_detail.meta['store'])
    if job.details and 'store' in job.details:
        store.update(job.details['store'])
    engine = engines.load_from_detail(flow_detail, store=store, engine=self._engine, backend=self._persistence, **self._engine_options)
    return engine