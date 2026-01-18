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
def DataAttributeBindingSetIamPolicyFromFile(attribute_binding_ref, policy_file):
    """Set IAM policy binding request from file."""
    policy = iam_util.ParsePolicyFile(policy_file, dataplex_api.GetMessageModule().GoogleIamV1Policy)
    return DataAttributeSetIamPolicy(attribute_binding_ref, policy)