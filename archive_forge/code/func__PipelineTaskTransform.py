from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def _PipelineTaskTransform(pipeline_task):
    """Transform pipeline task message."""
    if 'taskSpec' in pipeline_task:
        popped_task_spec = pipeline_task.pop('taskSpec')
        for param_spec in popped_task_spec.get('params', []):
            input_util.ParamSpecTransform(param_spec)
        pipeline_task['taskSpec'] = {}
        pipeline_task['taskSpec']['taskSpec'] = popped_task_spec
    elif 'taskRef' in pipeline_task:
        input_util.RefTransform(pipeline_task['taskRef'])
        pipeline_task['taskRef'] = pipeline_task.pop('taskRef')
    if 'when' in pipeline_task:
        for when_expression in pipeline_task.get('when', []):
            _WhenExpressionTransform(when_expression)
        pipeline_task['whenExpressions'] = pipeline_task.pop('when')
    input_util.ParamDictTransform(pipeline_task.get('params', []))