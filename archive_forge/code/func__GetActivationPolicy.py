from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetActivationPolicy(self, cp_type, policy):
    if policy is None:
        return cp_type.ActivationPolicyValueValuesEnum.SQL_ACTIVATION_POLICY_UNSPECIFIED
    return cp_type.ActivationPolicyValueValuesEnum.lookup_by_name(policy)