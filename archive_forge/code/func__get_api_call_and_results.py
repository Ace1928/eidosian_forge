import os
import json
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
import parlai.tasks.google_sgd.build as build_
def _get_api_call_and_results(self, sys_turn, schema_lookup):
    api_call = {}
    api_resp = {}
    for frame in sys_turn['frames']:
        if 'service_call' in frame:
            method = frame['service_call']['method']
            for slot_type, slot_value in frame['service_call']['parameters'].items():
                api_call[f'{method}.{slot_type}'] = slot_value
            assert 'service_results' in frame
        if 'actions' in frame:
            for action in frame['actions']:
                slot_type = action['slot']
                slot_value = action['canonical_values']
                api_resp[slot_type] = slot_value
    return (api_call, api_resp)