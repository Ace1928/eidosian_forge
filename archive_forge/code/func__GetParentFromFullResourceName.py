from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.bqexports import bqexport_util
from googlecloudsdk.command_lib.scc.bqexports import flags as bqexports_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetParentFromFullResourceName(config_name):
    """Returns the parts of the BigQuery export name before "/bigQueryExports"."""
    return '/'.join(config_name.split('/')[:-2])