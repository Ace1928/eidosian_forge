from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def ReformatExpiryDetail(version, expiration, command):
    """Reformat expiration string to ExpiryDetail object.

  Args:
    version: Release track information
    expiration: expiration string.
    command: gcloud command name.

  Returns:
    ExpiryDetail object that contains the expiration data.

  """
    messages = ci_client.GetMessages(version)
    duration = 'P' + expiration
    expiration_ts = FormatDateTime(duration)
    if version == 'v1alpha1' and command == 'modify-membership-roles':
        return messages.MembershipRoleExpiryDetail(expireTime=expiration_ts)
    return messages.ExpiryDetail(expireTime=expiration_ts)