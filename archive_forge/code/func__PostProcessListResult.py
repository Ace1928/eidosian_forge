from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import resource_args
from googlecloudsdk.core.resource import resource_projector
def _PostProcessListResult(results):
    results_json = resource_projector.MakeSerializable(results)
    for result in results_json:
        result['zone'] = result['name'].split('/')[3]
    return results_json