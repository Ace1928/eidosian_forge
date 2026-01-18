from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.apigee import base
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import request
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import log
@classmethod
def SplitName(cls, operation_info):
    name_parts = re.match('organizations/([a-z][-a-z0-9]{0,30}[a-z0-9])/operations/([0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})', operation_info['name'])
    if not name_parts:
        return operation_info
    operation_info['organization'] = name_parts.group(1)
    operation_info['uuid'] = name_parts.group(2)
    return operation_info