from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store as creds_store
from googlecloudsdk.core.util import files
def _get_existing_transfer_roles_for_account(project_iam_policy, prefixed_account_email, roles_set):
    """Returns roles in IAM policy from roles_set assigned to account email."""
    roles = set()
    for binding in project_iam_policy.bindings:
        if any([m == prefixed_account_email for m in binding.members]) and binding.role in roles_set:
            roles.add(binding.role)
    return roles