from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.backup_restore import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ProcessSelectedNamespaces(selected_namespaces):
    if not selected_namespaces:
        raise exceptions.InvalidArgumentException('--selected-namespaces', 'Selected namespaces must not be empty.')
    return selected_namespaces