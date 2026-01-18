from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def OrgPoliciesMessages():
    return apis.GetMessagesModule('cloudresourcemanager', ORG_POLICIES_API_VERSION)