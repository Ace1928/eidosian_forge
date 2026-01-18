from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.docker import credential_utils as cred_utils
from googlecloudsdk.core.util import files as file_utils
def DockerCredentialGcloudExists(self):
    return file_utils.SearchForExecutableOnPath('docker-credential-gcloud') or file_utils.SearchForExecutableOnPath('docker-credential-gcloud.cmd')