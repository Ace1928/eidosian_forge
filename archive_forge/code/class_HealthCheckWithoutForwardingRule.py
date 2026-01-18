from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from dns import rdatatype
from googlecloudsdk.api_lib.dns import import_util
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class HealthCheckWithoutForwardingRule(exceptions.Error):
    """Health check enabled but no forwarding rules present."""