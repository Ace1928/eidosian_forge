import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def _build_schedule(self, cron_trigger, wf_input):
    if isinstance(wf_input, str):
        wf_input = json.loads(wf_input)
    sched_info = {'id': cron_trigger.name, 'name': wf_input['name'], 'instance': wf_input['instance'], 'parent_id': wf_input.get('parent_id', None), 'created_at': cron_trigger.created_at, 'next_execution_time': cron_trigger.next_execution_time, 'pattern': cron_trigger.pattern, 'input': cron_trigger.workflow_input}
    if hasattr(cron_trigger, 'updated_at'):
        sched_info['updated_at'] = cron_trigger.updated_at
    return Schedule(self, sched_info, loaded=True)