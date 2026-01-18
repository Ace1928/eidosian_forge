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
def CheckRedirectionPermission(projects):
    """Checks redirection permission for the projects."""
    for project in projects:
        con = console_attr.GetConsoleAttr()
        authorized = ar_requests.TestRedirectionIAMPermission(project)
        if not authorized:
            if len(projects) == 1:
                log.status.Print(con.Colorize('FAIL: ', 'red') + f'This operation requires the {','.join(ar_requests.REDIRECT_PERMISSIONS)} permission(s) on project {project}.')
            else:
                log.status.Print(con.Colorize('FAIL: ', 'red') + f'This operation requires the {','.join(ar_requests.REDIRECT_PERMISSIONS)} permission(s) on each project to migrate, including {project}.')
            user = properties.VALUES.core.account.Get()
            if user.endswith('gserviceaccount.com'):
                prefix = 'serviceAccount'
            else:
                prefix = 'user'
            log.status.Print(f"You can set this permission with the following command:\n  gcloud projects add-iam-policy-binding {project} --member={prefix}:{user} --role='roles/storage.admin'")
            return False
    return True