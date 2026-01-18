from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def GetApplicationWorkloadRef(args):
    """Returns a application workload reference."""
    workload_ref = args.CONCEPTS.workload.Parse()
    if not workload_ref.Name():
        raise exceptions.InvalidArgumentException('workload', 'workload id must be non-empty.')
    return workload_ref