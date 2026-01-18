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
def CheckServiceAccountPermission(unused_repo_ref, repo_args, request):
    """Checks and grants key encrypt/decrypt permission for service account.

  Checks if Artifact Registry service account has encrypter/decrypter or owner
  role for the given key. If not, prompts users to grant key encrypter/decrypter
  permission to the service account. Operation would fail if users do not grant
  the permission.

  Args:
    unused_repo_ref: Repo reference input.
    repo_args: User input arguments.
    request: Create repository request.

  Returns:
    Create repository request.
  """
    if not repo_args.kms_key:
        return request
    try:
        project_num = project_util.GetProjectNumber(GetProject(repo_args))
        policy = ar_requests.GetCryptoKeyPolicy(repo_args.kms_key)
        service_account = _AR_SERVICE_ACCOUNT.format(project_num=project_num)
        for binding in policy.bindings:
            if 'serviceAccount:' + service_account in binding.members and (binding.role == 'roles/cloudkms.cryptoKeyEncrypterDecrypter' or binding.role == 'roles/owner'):
                return request
        grant_permission = console_io.PromptContinue(prompt_string='\nGrant the Artifact Registry Service Account permission to encrypt/decrypt with the selected key [{key_name}]'.format(key_name=repo_args.kms_key))
        if not grant_permission:
            return request
        try:
            ar_requests.AddCryptoKeyPermission(repo_args.kms_key, 'serviceAccount:' + service_account)
        except apitools_exceptions.HttpBadRequestError:
            msg = 'The Artifact Registry service account might not exist, manually create the service account.\nLearn more: https://cloud.google.com/artifact-registry/docs/cmek'
            raise ar_exceptions.ArtifactRegistryError(msg)
        log.status.Print('Added Cloud KMS CryptoKey Encrypter/Decrypter Role to [{key_name}]'.format(key_name=repo_args.kms_key))
    except apitools_exceptions.HttpForbiddenError:
        return request
    return request