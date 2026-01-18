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
def CopyImagesFromGCR(thread_futures, executor, repo_path, recent_images, last_uploaded, copy_from, results):
    """Recursively copies images from GCR."""
    http_obj = util.Http(timeout=10 * 60)
    repository = docker_name.Repository(repo_path)
    next_page = ''
    while True:
        try:
            with docker_image.FromRegistry(basic_creds=util.CredentialProvider(), name=repository, transport=http_obj) as image:
                query = f'?CopyFromGCR={copy_from}'
                if recent_images:
                    query += f'&PullDays={recent_images}'
                if last_uploaded:
                    query += f'&MaxVersions={last_uploaded}'
                if next_page:
                    query += f'&NextPage={next_page}'
                tags_payload = json.loads(image._content(f'tags/list{query}').decode('utf8'))
                if tags_payload.get('nextPage'):
                    next_page = tags_payload['nextPage']
                else:
                    break
        except requests.exceptions.ReadTimeout:
            continue
    results['manifestsCopied'] += tags_payload.get('manifestsCopied', 0)
    results['tagsCopied'] += tags_payload.get('tagsCopied', 0)
    results['manifestsFailed'] += tags_payload.get('manifestsFailed', 0)
    results['tagsFailed'] += tags_payload.get('tagsFailed', 0)
    failures = tags_payload.get('exampleFailures', [])
    if failures:
        if not results['example_failures']:
            results['example_failures'] = []
        results['example_failures'] = (results['example_failures'] + failures)[0:10]
    for child in tags_payload['child']:
        copy_args = [thread_futures, executor, repo_path + '/' + child, recent_images, last_uploaded, copy_from, results]
        if executor:
            thread_futures.append(executor.submit(CopyImagesFromGCR, *copy_args))
        else:
            CopyImagesFromGCR(*copy_args)
    return results