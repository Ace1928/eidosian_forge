from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import resource_args
from googlecloudsdk.core.resource import resource_projector
def _TransformFailureReason(resource):
    max_len = 80
    message = resource.get('failureReason', '')
    return message[:max_len] + '..' if len(message) > max_len else message