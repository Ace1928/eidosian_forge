from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def GetApplicationServiceRef(args):
    """Returns a application service reference."""
    service_ref = args.CONCEPTS.service.Parse()
    if not service_ref.Name():
        raise exceptions.InvalidArgumentException('service', 'service id must be non-empty.')
    return service_ref