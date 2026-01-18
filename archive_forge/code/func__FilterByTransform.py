from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.command_lib.dataflow import job_utils
from googlecloudsdk.core.util import times
def _FilterByTransform(self, metric, transform):
    output_user_name = self._GetContextValue(metric, 'output_user_name') or ''
    step = self._GetContextValue(metric, 'step') or ''
    transform = re.compile(transform or '')
    if transform.match(output_user_name) or transform.match(step):
        return True
    return False