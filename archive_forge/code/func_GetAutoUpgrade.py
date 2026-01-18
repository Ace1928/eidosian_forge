from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def GetAutoUpgrade(args):
    """Gets the value of node auto-upgrade."""
    if args.IsSpecified('enable_autoupgrade'):
        return args.enable_autoupgrade
    if getattr(args, 'enable_kubernetes_alpha', False):
        return None
    return args.enable_autoupgrade