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
def RecommendAuthChange(policy_addition, existing_policy, location, project, repo, failures, pkg_dev=False):
    """Prompts the user to possibly change the repository's iam policy."""
    con = console_attr.GetConsoleAttr()
    log.status.Print(con.Emphasize('\nPotential IAM change for {} repository in project {}:\n'.format(repo, project), bold=True))
    if existing_policy.bindings:
        etag = existing_policy.etag
        existing_policy_bindings = encoding.MessageToDict(existing_policy)['bindings']
        existing_string = yaml.dump({'bindings': existing_policy_bindings})
        new_string = yaml.dump(encoding.MessageToDict(policy_addition)).split('\n', 1)[1]
        string_policy = '# Existing repository policy:\n{existing}\n# New additions:\n{new}'.format(existing=existing_string, new=new_string)
    else:
        string_policy = yaml.dump(encoding.MessageToDict(policy_addition))
        etag = ''
    log.status.Print(string_policy)
    message = 'This IAM policy will grant users the ability to perform all actions in Artifact Registry that they can currently perform in Container Registry. This policy may allow access that was previously prevented by deny policies or IAM conditions.'
    if failures:
        message += f'\n\n{con.Colorize('Warning:', 'red')} Generated bindings may be insufficient because you do not have access to analyze IAM for the following resources: {failures}\nSee https://cloud.google.com/policy-intelligence/docs/analyze-iam-policies#required-permissions\n\n'
    if not console_io.CanPrompt():
        log.status.Print(message)
        log.status.Print('\nPrompting is disabled. To make interactive iam changes, enable prompting. Otherwise, manually add any missing Artifact Registry permissions and rerun using --skip-iam-update.')
    edited = False
    while True:
        options = ['Apply {} policy to the {}/{} Artifact Registry repository'.format('edited' if edited else 'above', project, repo), 'Edit policy']
        if pkg_dev:
            options.append('Do not change permissions for this repo')
            choices = ['apply', 'edit', 'skip', 'exit']
        else:
            options.append(f'Do not change permissions for this repo (users may lose access to {repo}/{project.replace(':', '/')})')
            options.append('Skip permission updates for all remaining repos (users may lose access to all remaining repos)')
            choices = ['apply', 'edit', 'skip', 'skip_all', 'exit']
        options.append('Exit')
        option = console_io.PromptChoice(message=message, options=options, default=1)
        if option < 0 or option >= len(choices):
            raise ValueError(f'Unknown option: {option}')
        if choices[option] == 'apply':
            log.status.Print('Applying policy to repository {}/{}'.format(project, repo))
            new_binding = encoding.PyValueToMessage(ar_requests.GetMessages().Policy, yaml.load(string_policy))
            if etag:
                new_binding.etag = etag
            try:
                ar_requests.SetIamPolicy('projects/{}/locations/{}/repositories/{}'.format(project, location, repo), new_binding)
                return True
            except apitools_exceptions.HttpError as e:
                log.status.Print('\nFailed to update iam policy:\n{}\n'.format(json.loads(e.content)['error']['message']))
        elif choices[option] == 'edit':
            string_policy = edit.OnlineEdit(string_policy)
            message = con.Emphasize('\nEdited policy:', bold=True) + '\n\n{}\n'.format(string_policy)
            edited = True
            continue
        elif choices[option] == 'skip':
            return True
        elif choices[option] == 'skip_all':
            return False
        elif choices[option] == 'exit':
            raise console_io.OperationCancelledError()
        else:
            raise ValueError(f'Unknown choice: {choices[option]}')