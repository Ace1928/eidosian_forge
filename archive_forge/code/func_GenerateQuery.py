from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def GenerateQuery(unused_ref, args, request):
    """Generate and set the query on the request based on the args.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    customer_id = GetCustomerId(args)
    labels = FilterLabels(args.labels)
    labels_str = ','.join(labels)
    request.query = 'parent=="{0}" && "{1}" in labels'.format(customer_id, labels_str)
    return request