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
def _get_iam_prefixed_email(email_string, is_service_account):
    """Returns an email format useful for interacting with IAM APIs."""
    iam_prefix = 'serviceAccount' if is_service_account else 'user'
    return '{}:{}'.format(iam_prefix, email_string)