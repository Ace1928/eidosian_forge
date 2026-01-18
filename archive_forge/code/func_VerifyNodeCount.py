from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def VerifyNodeCount(ref, args, request):
    del ref
    if args.node_count and args.node_count <= 1:
        raise exceptions.InvalidArgumentException('--node-count', 'Node count must be greater than 1')
    return request