from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def _GetLabelAttributes(log_entry):
    """Reads the label attributes of the given log entry."""
    label_attributes = {'task_name': 'unknown_task'}
    labels = _ToDict(log_entry.labels)
    resource_labels = {} if not log_entry.resource else _ToDict(log_entry.resource.labels)
    if resource_labels.get('task_name') is not None:
        label_attributes['task_name'] = resource_labels['task_name']
    elif labels.get('task_name') is not None:
        label_attributes['task_name'] = labels['task_name']
    elif labels.get('ml.googleapis.com/task_name') is not None:
        label_attributes['task_name'] = labels['ml.googleapis.com/task_name']
    if labels.get('trial_id') is not None:
        label_attributes['trial_id'] = labels['trial_id']
    elif labels.get('ml.googleapis.com/trial_id') is not None:
        label_attributes['trial_id'] = labels['ml.googleapis.com/trial_id']
    return label_attributes