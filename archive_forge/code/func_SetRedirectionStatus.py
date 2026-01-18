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
def SetRedirectionStatus(project, status, pull_percent=None):
    """Sets the redirection status for the given project."""
    endpoint_property = getattr(properties.VALUES.api_endpoint_overrides, 'artifactregistry')
    old_endpoint = endpoint_property.Get()
    env = 'prod'
    try:
        if old_endpoint and 'staging' in old_endpoint:
            env = 'staging'
            endpoint_property.Set('https://artifactregistry.googleapis.com/')
        ar_requests.SetUpgradeRedirectionState(project, status, pull_percent)
    except apitools_exceptions.HttpForbiddenError as e:
        con = console_attr.GetConsoleAttr()
        match = re.search('requires (.*) to have storage.objects.', str(e))
        if not match:
            raise
        log.status.Print(con.Colorize('\nERROR:', 'red') + " The Artifact Registry service account doesn't have access to {project} for copying images\nThe following command will grant the necessary access (may take a few minutes):\n  gcloud projects add-iam-policy-binding {project} --member='serviceAccount:{p4sa}' --role='roles/storage.objectViewer'\n".format(p4sa=match[1], project=project))
        return False
    finally:
        if env == 'staging':
            endpoint_property.Set(old_endpoint)
    return True