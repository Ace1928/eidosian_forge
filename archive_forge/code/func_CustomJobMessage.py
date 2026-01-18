from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core.console import console_io
def CustomJobMessage(self):
    """Retures the CustomJob resource message."""
    return self.GetMessage('CustomJob')