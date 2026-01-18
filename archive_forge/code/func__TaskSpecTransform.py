from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import log
def _TaskSpecTransform(spec):
    for param_spec in spec.get('params', []):
        input_util.ParamSpecTransform(param_spec)
    for task_result in spec.get('results', []):
        input_util.TaskResultTransform(task_result)