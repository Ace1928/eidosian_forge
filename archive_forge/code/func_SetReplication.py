from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
def SetReplication(self, secret_ref, policy, locations, keys):
    """Set the replication policy on an existing secret.."""
    replication = _MakeReplicationMessage(self.messages, policy, locations, keys)
    return self.service.Patch(self.messages.SecretmanagerProjectsSecretsPatchRequest(name=secret_ref.RelativeName(), secret=self.messages.Secret(replication=replication), updateMask=_FormatUpdateMask(['replication'])))