from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _RemoveBindingFromIamPolicyAllConditions(policy, member, role):
    """Remove all member/role bindings from policy regardless of condition."""
    conditions_removed = False
    for binding in policy.bindings:
        if role == binding.role and member in binding.members:
            binding.members.remove(member)
            conditions_removed = True
    if not conditions_removed:
        raise IamPolicyBindingNotFound('Policy bindings with the specified principal and role not found!')
    policy.bindings[:] = [b for b in policy.bindings if b.members]