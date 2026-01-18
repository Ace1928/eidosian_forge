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
def _user_cluster_id(self, args: parser_extensions.Namespace):
    """Parses user cluster from args and returns its ID."""
    user_cluster_ref = self._user_cluster_ref(args)
    if user_cluster_ref:
        return user_cluster_ref.Name()
    return None