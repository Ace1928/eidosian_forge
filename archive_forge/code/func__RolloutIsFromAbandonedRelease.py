from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _RolloutIsFromAbandonedRelease(rollout_obj):
    rollout_ref = RolloutReferenceFromName(rollout_obj.name)
    release_ref = rollout_ref.Parent()
    try:
        release_obj = release.ReleaseClient().Get(release_ref.RelativeName())
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)
    return release_obj.abandoned