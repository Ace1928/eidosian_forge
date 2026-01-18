import os
import asyncio
import logging
import time
from typing import Union
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import aiohttp.web
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray.dashboard.modules.event.event_utils import (
from ray.core.generated import event_pb2
from ray.core.generated import event_pb2_grpc
from ray.dashboard.datacenter import DataSource
@staticmethod
def _update_events(event_list):
    all_job_events = defaultdict(JobEvents)
    for event in event_list:
        event_id = event['event_id']
        custom_fields = event.get('custom_fields')
        system_event = False
        if custom_fields:
            job_id = custom_fields.get('job_id', 'global') or 'global'
        else:
            job_id = 'global'
        if system_event is False:
            all_job_events[job_id][event_id] = event
    for job_id, new_job_events in all_job_events.items():
        job_events = DataSource.events.get(job_id, JobEvents())
        job_events.update(new_job_events)
        DataSource.events[job_id] = job_events
        events = DataSource.events[job_id]
        if len(events) > MAX_EVENTS_TO_CACHE * 1.1:
            while len(events) > MAX_EVENTS_TO_CACHE:
                events.popitem(last=False)