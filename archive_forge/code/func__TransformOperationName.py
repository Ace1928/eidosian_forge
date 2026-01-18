from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _TransformOperationName(resource):
    """Get operation name without project prefix."""
    operation_name = resource.get('name')
    results = operation_name.split('/')
    short_name = '/'.join(results[3:])
    return short_name