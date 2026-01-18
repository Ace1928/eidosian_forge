from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def GetManagement(self):
    try:
        management = self[self.MANAGEMENT_KEY]
    except KeyError:
        return None
    return management