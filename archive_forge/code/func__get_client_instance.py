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
def _get_client_instance(self) -> client.GkeonpremV1:
    """Returns the client instance.

    Function created for IDE type hint only, inline type annotation is not
    supported due to gcloud using a low Python3 version.
    """
    return apis.GetClientInstance('gkeonprem', 'v1')