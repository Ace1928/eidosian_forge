from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def _EntryToDict(log_entry):
    """Converts a log entry to a dictionary."""
    output = {}
    output['severity'] = log_entry.severity.name if log_entry.severity else 'DEFAULT'
    output['timestamp'] = log_entry.timestamp
    label_attributes = _GetLabelAttributes(log_entry)
    output['task_name'] = label_attributes['task_name']
    if 'trial_id' in label_attributes:
        output['trial_id'] = label_attributes['trial_id']
    output['message'] = ''
    if log_entry.jsonPayload is not None:
        json_data = _ToDict(log_entry.jsonPayload)
        if 'message' in json_data:
            if json_data['message']:
                output['message'] += json_data['message']
            del json_data['message']
        if 'levelname' in json_data:
            del json_data['levelname']
        output['json'] = json_data
    elif log_entry.textPayload is not None:
        output['message'] += six.text_type(log_entry.textPayload)
    elif log_entry.protoPayload is not None:
        output['json'] = encoding.MessageToDict(log_entry.protoPayload)
    return output