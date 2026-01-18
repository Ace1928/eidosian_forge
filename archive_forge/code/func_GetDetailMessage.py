from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def GetDetailMessage(self):
    return self.status_detail