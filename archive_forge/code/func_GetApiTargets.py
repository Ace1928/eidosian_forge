from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base as calliope_base
def GetApiTargets(args, messages):
    """Create list of target apis."""
    api_targets = []
    for api_target in getattr(args, 'api_target', []) or []:
        api_targets.append(messages.V2ApiTarget(service=api_target.get('service'), methods=api_target.get('methods', [])))
    return api_targets