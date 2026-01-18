from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.core.console import console_io
def _GetAbandonReleaseMessageString(deployed_targets):
    if deployed_targets:
        return 'This release is the latest in {} target(s):\n{}\n'.format(len(deployed_targets), '\n'.join(['- {}'.format(target_ref.RelativeName()) for target_ref in deployed_targets]))
    return None