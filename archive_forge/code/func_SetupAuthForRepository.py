from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from concurrent import futures
import encodings.idna  # pylint: disable=unused-import
import json
import mimetypes
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib import artifacts
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import remote_repo_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import upgrade_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import edit
from googlecloudsdk.core.util import parallel
import requests
def SetupAuthForRepository(gcr_project, ar_project, host, repo, has_bucket, pkg_dev=False):
    """Checks permissions for a repository and prompts for changes if any is missing.

  Checks permission for a repository and provides a list of users/roles that had
  permissions in GCR but are missing equivalent roles in AR. Prompts the user to
  add these roles, edit them, or keep permissions the same.

  Args:
    gcr_project: The GCR project
    ar_project: The AR project
    host: The GCR host (like gcr.io)
    repo: The AR repo being copied to
    has_bucket: Whether a GCR bucket exists for this repository
    pkg_dev: If true, this is for a single pkg.dev repo (prompts are different)

  Returns:
    A tuple of (diffs_found, should_continue) where diffs_found is true if
    there were auth diffs found between GCR + AR and should_continue is true
    if the tool should continue recommending auth changes for subsequent
    repos.
  """
    gcr_auth, failures = upgrade_util.iam_map(host, gcr_project, skip_bucket=not has_bucket, from_ar_permissions=False, best_effort=True)
    if not gcr_auth and failures:
        WarnNoAuthGenerated(pkg_dev=pkg_dev)
        return (True, False)
    ar_non_repo_auth, _ = upgrade_util.iam_map('', ar_project, skip_bucket=True, from_ar_permissions=True, best_effort=True)
    ar_repo_policy = ar_requests.GetIamPolicy('projects/{}/locations/{}/repositories/{}'.format(ar_project, repo['location'], repo['repository']))
    missing_auth = CalculateMissingAuth(gcr_auth, ar_non_repo_auth, ar_repo_policy)
    if missing_auth:
        continue_checking_auth = RecommendAuthChange(upgrade_util.policy_from_map(missing_auth), ar_repo_policy, repo['location'], ar_project, repo['repository'], failures=failures, pkg_dev=pkg_dev)
        return (True, continue_checking_auth)
    return (False, True)