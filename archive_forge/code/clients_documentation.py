from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
List the details of the resident and descendant ETD effective custom modules.