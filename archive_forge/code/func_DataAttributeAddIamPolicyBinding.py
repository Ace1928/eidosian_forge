from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def DataAttributeAddIamPolicyBinding(data_attribute_ref, member, role):
    """Add IAM policy binding request."""
    policy = DataAttributeGetIamPolicy(data_attribute_ref)
    iam_util.AddBindingToIamPolicy(dataplex_api.GetMessageModule().GoogleIamV1Binding, policy, member, role)
    return DataAttributeSetIamPolicy(data_attribute_ref, policy)