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
def _GetContextValue(self, metric, key):
    if metric.name.context:
        for prop in metric.name.context.additionalProperties:
            if prop.key == key:
                return prop.value
    return None