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
class EnvironmentsClient(base.BaseClient):
    _entity_path = ['organization', 'environment']