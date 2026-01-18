from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.run.printers import revision_printer
from googlecloudsdk.command_lib.run.printers import traffic_printer
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _GetRevisionHeader(self, record):
    header = ''
    if record.status is None:
        header = 'Unknown revision'
    else:
        header = 'Revision {}'.format(record.status.latestCreatedRevisionName)
    return console_attr.GetConsoleAttr().Emphasize(header)