from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_client as client
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
import six
def _standalone_cluster_id(self, args: parser_extensions.Namespace):
    """Parses standalone cluster from the given args and returns its ID."""
    standalone_cluster_ref = self._standalone_cluster_ref(args)
    if standalone_cluster_ref:
        return standalone_cluster_ref.Name()
    return None