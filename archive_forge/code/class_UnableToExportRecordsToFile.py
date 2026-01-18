from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dns import name
from dns import rdata
from dns import rdataclass
from dns import rdatatype
from dns import zone
from googlecloudsdk.api_lib.dns import svcb_stub
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_printer
class UnableToExportRecordsToFile(Error):
    """Unable to export records to specified file."""