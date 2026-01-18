import asyncio
import logging
import os
import time
from collections import deque
import aiohttp.web
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.gcs_pubsub import GcsAioActorSubscriber
from ray.core.generated import (
from ray.dashboard.datacenter import DataSource, DataOrganizer
from ray.dashboard.modules.actor import actor_consts
from ray.dashboard.optional_utils import rest_response
def actor_table_data_to_dict(message):
    orig_message = dashboard_utils.message_to_dict(message, {'actorId', 'parentId', 'jobId', 'workerId', 'rayletId', 'callerId', 'taskId', 'parentTaskId', 'sourceActorId'}, always_print_fields_with_no_presence=True)
    fields = {'actorId', 'jobId', 'pid', 'address', 'state', 'name', 'numRestarts', 'timestamp', 'className', 'startTime', 'endTime', 'reprName'}
    light_message = {k: v for k, v in orig_message.items() if k in fields}
    light_message['actorClass'] = orig_message['className']
    exit_detail = '-'
    if 'deathCause' in orig_message:
        context = orig_message['deathCause']
        if 'actorDiedErrorContext' in context:
            exit_detail = context['actorDiedErrorContext']['errorMessage']
        elif 'runtimeEnvFailedContext' in context:
            exit_detail = context['runtimeEnvFailedContext']['errorMessage']
        elif 'actorUnschedulableContext' in context:
            exit_detail = context['actorUnschedulableContext']['errorMessage']
        elif 'creationTaskFailureContext' in context:
            exit_detail = context['creationTaskFailureContext']['formattedExceptionString']
    light_message['exitDetail'] = exit_detail
    light_message['startTime'] = int(light_message['startTime'])
    light_message['endTime'] = int(light_message['endTime'])
    light_message['requiredResources'] = dict(message.required_resources)
    return light_message