from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.backup_restore import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def PreprocessUpdateBackupPlan(ref, args, request):
    """Preprocesses request and update mask for backup update command."""
    del ref
    if args.IsSpecified('selected_namespaces'):
        request.backupPlan.backupConfig.selectedApplications = None
        request.backupPlan.backupConfig.allNamespaces = None
    if args.IsSpecified('selected_applications'):
        request.backupPlan.backupConfig.selectedNamespaces = None
        request.backupPlan.backupConfig.allNamespaces = None
    if args.IsSpecified('all_namespaces'):
        request.backupPlan.backupConfig.selectedApplications = None
        request.backupPlan.backupConfig.selectedNamespaces = None
    new_masks = []
    for mask in request.updateMask.split(','):
        if mask.startswith('backupConfig.selectedNamespaces'):
            mask = 'backupConfig.selectedNamespaces'
        elif mask.startswith('backupConfig.selectedApplications'):
            mask = 'backupConfig.selectedApplications'
        new_masks.append(mask)
    request.updateMask = ','.join(new_masks)
    return request