from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def ParseOSConfigAssignmentFile(ref, args, req):
    """Returns modified request with parsed OS policy assignment and update mask."""
    del ref
    api_version = GetApiVersion(args)
    messages = GetApiMessage(api_version)
    policy_assignment_config, update_fields = GetResourceAndUpdateFieldsFromFile(args.file, messages.OSPolicyAssignment)
    req.oSPolicyAssignment = policy_assignment_config
    update_fields.sort()
    if 'update' in args.command_path:
        req.updateMask = ','.join(update_fields)
    return req