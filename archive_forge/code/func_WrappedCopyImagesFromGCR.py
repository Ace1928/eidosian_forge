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
def WrappedCopyImagesFromGCR(hosts, project_repo, recent_images, last_uploaded, copy_from='same', convert_to_pkg_dev=False, max_threads=8, pre_copy=False):
    """Copies images from GCR for all hosts and handles auth error."""
    original_project_repo = project_repo
    project_repo = project_repo.replace(':', '/')
    try:
        results = collections.defaultdict(int)
        if copy_from == 'same':
            if len(hosts) == 1:
                message = f'Copying images for {hosts[0]}/{project_repo}... '
            else:
                message = f'Copying images for {project_repo}... '
        elif len(hosts) == 1:
            message = f'Copying images to {hosts[0]}/{project_repo}... '
        else:
            message = f'Copying images to {project_repo}... '
        with progress_tracker.ProgressTracker(message, tick_delay=2, no_spacing=True):
            with futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                thread_futures = []
                for host in sorted(hosts):
                    if convert_to_pkg_dev:
                        endpoint_prefix = properties.VALUES.artifacts.registry_endpoint_prefix.Get()
                        location = _ALLOWED_GCR_REPO_LOCATION[host]
                        url = f'{endpoint_prefix}{location}-docker.pkg.dev/{project_repo}/{host}'
                    else:
                        url = f'{host}/{project_repo}'
                    copy_args = [thread_futures, executor if max_threads > 1 else None, url, recent_images, last_uploaded, copy_from, results]
                    if max_threads > 1:
                        thread_futures.append(executor.submit(CopyImagesFromGCR, *copy_args))
                    else:
                        CopyImagesFromGCR(*copy_args)
                while thread_futures:
                    future = thread_futures.pop()
                    future.result()
        log.status.Print('\n{project}: Successfully copied {tags} additional tags and {manifests} additional manifests. There were {failures} failures.'.format(project=project_repo, tags=results['tagsCopied'], manifests=results['manifestsCopied'], failures=results['tagsFailed'] + results['manifestsFailed']))
        if results['tagsFailed'] + results['manifestsFailed']:
            log.status.Print('\nExample images that failed to copy:')
            for example_failure in results['example_failures']:
                log.status.Print(example_failure)
            return pre_copy
        return True
    except docker_http.V2DiagnosticException as e:
        match = re.search('requires (.*) to have storage.objects.', str(e))
        if not match:
            raise
        con = console_attr.GetConsoleAttr()
        project = original_project_repo
        if copy_from != 'same':
            project = copy_from.split('/')[-1]
        log.status.Print(con.Colorize('\nERROR:', 'red') + f" The Artifact Registry service account doesn't have access to {project} for copying images\nThe following command will grant the necessary access (may take a few minutes):\n  gcloud projects add-iam-policy-binding {project} --member='serviceAccount:{match[1]}' --role='roles/storage.objectViewer'\nYou can re-run this script after granting access.")
        return False