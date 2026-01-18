from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.backup_restore import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ParseGroupKinds(group_kinds, flag='--cluster-resource-restore-scope'):
    """Process list of group kinds."""
    if not group_kinds:
        return None
    message = api_util.GetMessagesModule()
    gks = []
    try:
        for resource in group_kinds:
            group_kind = resource.split('/')
            if len(group_kind) == 1:
                group = ''
                kind = group_kind[0]
            elif len(group_kind) == 2:
                group, kind = group_kind
            else:
                raise exceptions.InvalidArgumentException(flag, 'Cluster resource restore scope is invalid.')
            if not kind:
                raise exceptions.InvalidArgumentException(flag, 'Cluster resource restore scope kind is empty.')
            gk = message.GroupKind()
            gk.resourceGroup = group
            gk.resourceKind = kind
            gks.append(gk)
        return gks
    except ValueError:
        raise exceptions.InvalidArgumentException(flag, 'Cluster resource restore scope is invalid.')