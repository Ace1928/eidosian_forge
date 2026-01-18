from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.metastore import operations_util as operations_api_util
from googlecloudsdk.api_lib.metastore import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
class _PendingFederationDelete(object):
    """Data class holding information about a pending federation deletion."""

    def __init__(self, federation_name, operation):
        self.federation_name = federation_name
        self.operation = operation