from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import log
def _TaskTransform(task):
    """Transform task message."""
    if 'taskSpec' in task:
        task_spec = task.pop('taskSpec')
        _TaskSpecTransform(task_spec)
        managed_sidecars = _MetadataToSidecar(task_spec.pop('metadata')) if 'metadata' in task_spec else []
        if managed_sidecars:
            task_spec['managedSidecars'] = managed_sidecars
        task['taskSpec'] = {'taskSpec': task_spec}
    if 'taskRef' in task:
        input_util.RefTransform(task['taskRef'])
    whens = task.pop('when', [])
    for when in whens:
        if 'operator' in when:
            when['expressionOperator'] = input_util.CamelToSnake(when.pop('operator')).upper()
    task['whenExpressions'] = whens
    input_util.ParamDictTransform(task.get('params', []))